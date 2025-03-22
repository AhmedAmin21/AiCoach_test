from langchain.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import requests
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.callbacks import StreamlitCallbackHandler
from deep_translator import GoogleTranslator

load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
ninga_api_key = os.getenv('NINGA_API_KEY')

api_url = 'https://api.calorieninjas.com/v1/nutrition?query='

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def detect_language(text):
    return GoogleTranslator(source='auto', target='en').translate(text) != text

def Ninga(query: str):
    response = requests.get(api_url + query, headers={'X-Api-Key': ninga_api_key})
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            return {
                "name": item.get("name", "Unknown"),
                "serving_size": item.get("serving_size_g", "N/A"),
                "calories": item.get("calories", "N/A"),
                "total_fat": item.get("fat_total_g", "N/A"),
                "protein": item.get("protein_g", "N/A"),
                "sugar": item.get("sugar_g", "N/A")
            }
        else:
            return "No nutrition data found."
    else:
        return f"API request failed with status code {response.status_code}"

api_calling_tool = StructuredTool.from_function(
    func=Ninga,
    name='Ninga',
    description='Returns the nutrition info of a food item (name, calories, fat, protein, sugar, serving size).'
)

tools = [api_calling_tool]

llm_nutrition = ChatGroq(api_key=api_key, model_name='gemma2-9b-it')

nutrition_prompt = ChatPromptTemplate.from_messages([
    ('system', 
      """You are a nutritionist and you have two primary goals:
     1. If the user asks about a food item, always provide its nutritional details first by calling the API.
     2. If the user asks to replace an item, first find an alternative food that has a similar nutritional profile give two in minimum.
     3. Then, compare the new food with the original item based on nutrition data.
     
     Always format your answer like this:
     - **Replacement Food:** [Food Name]
     - **Comparison:**
         - **Original:** [Calories, Protein, Fat, etc.]
         - **Replacement:** [Calories, Protein, Fat, etc.]

     Never make assumptions or provide information without calling the API first.
     """),      
    ('human', '{input}'),
    ("placeholder", "{agent_scratchpad}")
])

tool_agent = create_tool_calling_agent(llm_nutrition, tools, nutrition_prompt)
agent_ex = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)

llm_chatbot = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", api_key="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")

template = """You are a fitness coach with 10 years of experience. You are an expert in both nutrition and gym training.
You also have knowledge of human anatomy, body parts, and how to target them in weightlifting training.
You can answer fitness/nutrition questions in **both Arabic and English**.  
**Important Rules:**
- Always reply in the **same language** as the user's question.
- If the question is related to nutrition, extract and explain the nutritional data.
- Keep responses **concise yet informative**.
question:
{question}"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=template
)

sop = StrOutputParser()
fitness_chain = prompt_template | llm_chatbot | sop

st.title('Fitness & Nutrition Chatbot')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': "Hi! I'm a fitness coach. I can answer gym or nutrition questions."}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if user_prompt := st.chat_input(placeholder='What is the right number of sets should I play for a {specific training}?'):
    st.session_state.messages.append({'role': 'user', 'content': user_prompt})
    st.chat_message("user").write(user_prompt)

    user_lang = 'ar' if detect_language(user_prompt) else 'en'
    translated_prompt = translate_text(user_prompt, 'en') if user_lang == 'ar' else user_prompt

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

    nutrition_keywords = {
    "calories", "protein", "fat", "sugar", "nutrition", "food", "diet",
    "rice", "wheat", "oats", "corn", "barley", "quinoa", "millet", "rye",
    "spinach", "lettuce", "kale", "carrot", "potato", "beet", 
    "broccoli", "cauliflower", "cabbage", "tomato", "bell pepper", "eggplant",
    "peas", "green beans", "zucchini", "pumpkin",
    "orange", "lemon", "grapefruit", "strawberry", "blueberry", "raspberry",
    "banana", "mango", "pineapple", "peach", "cherry", "plum",
    "watermelon", "cantaloupe", "apple", "pear", "grape",
    "beef", "pork", "lamb", "chicken", "turkey", "duck",
    "salmon", "tuna", "shrimp", "egg", 'meat',
    "lentils", "chickpeas", "black beans", "kidney beans", 
    "almonds", "walnuts", "sunflower seeds", "tofu", "tempeh", "edamame",
    "milk", "goat milk", "cheddar cheese", "mozzarella cheese", "feta cheese",
    "yogurt", "butter", "almond milk", "soy milk", "coconut yogurt",
    "ghee", "olive oil", "coconut oil", "sunflower oil", "canola oil", "soybean oil",
    "flaxseeds", "chia seeds", "peanuts", "avocado",
    "basil", "cilantro", "parsley", "garlic", "onion", "ginger",
    "black pepper", "cumin", "paprika", "turmeric", "cinnamon", "cloves",
    "sugar", "brown sugar", "honey", "maple syrup", "molasses",
    "tea", "coffee", "orange juice"
    }

    user_prompt_lower = translated_prompt.lower()

    if any(keyword in user_prompt_lower for keyword in nutrition_keywords):
        answer_ = agent_ex.invoke({'input': translated_prompt})
        answer = fitness_chain.invoke({'question': answer_['output']})
    else:
        answer = fitness_chain.invoke({'question': translated_prompt})

    final_answer = translate_text(answer, 'ar') if user_lang == 'ar' else answer
    st.session_state.messages.append({'role': 'assistant', 'content': final_answer})
    if user_lang == 'ar':
        st.markdown(f'<div dir="rtl" style="text-align: right;">{final_answer}</div>', unsafe_allow_html=True)
    else:
        st.write(final_answer)
    #st.write(final_answer)
