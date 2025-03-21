from langchain.tools import StructuredTool
from langchain.chains import LLMChain
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




load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
ninga_api_key = os.getenv('NINGA_API_KEY')

# API URL for nutrition lookup
api_url = 'https://api.calorieninjas.com/v1/nutrition?query='

# Function to fetch nutrition data
def Ninga(query: str):
    response = requests.get(api_url + query, headers={'X-Api-Key': ninga_api_key})
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]  # Get first food item
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

# Create nutrition tool
api_calling_tool = StructuredTool.from_function(
    func=Ninga,
    name='Ninga',
    description='Returns the nutrition info of a food item (name, calories, fat, protein, sugar, serving size).'
)

tools = [api_calling_tool]

# LLM for nutrition agent
llm_nutrition = ChatGroq(api_key=api_key, model_name='gemma2-9b-it')

# Prompt for nutrition agent
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

# Create tool agent for nutrition lookup
tool_agent = create_tool_calling_agent(llm_nutrition, tools, nutrition_prompt)
agent_ex = AgentExecutor(agent=tool_agent, tools=tools, verbose=True)


#llm = ChatGroq(model='gemma2-9b-it', api_key=api_key)
llm_chatbot = ChatNVIDIA( model="meta/llama-3.3-70b-instruct",
  api_key="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")

template = """you are a fitness coach with a 10 years experaince, you have an expert knowledge in both nutrition and gym training, 
you have knowledge in human anatomy , you know every body parts and how to target them in weight lifting training.
your goal is to answer user fitness/nutrition question correctly with a nice and easy way to understand and keep the answer short but containes the most important parts.
if you get an input as a nutrition facts take the info and provide it to user to help him understand the food he is eating.
question:
{question}"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=template
)

sop = StrOutputParser()
fitness_chain = prompt_template | llm_chatbot | sop
# -----------------------------------------------

# Streamlit UI
st.title('Fitness & Nutrition Chatbot')

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': "Hi! I'm a fitness coach. I can answer gym or nutrition questions."}
    ]

# write the message to user
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# 1 prompt = text input its place holder is 'What is machine learning?'
if user_prompt:=st.chat_input(placeholder='what is the right number of sets should i play for a {spacific training} '):
    st.session_state.messages.append({'role':'user', 'content':user_prompt})
    st.chat_message("user").write(user_prompt)


    with st.chat_message('assistant'):
        # here we show how agent think before giving the last answer
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

    # Check if the question is related to nutrition
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
    "water", "tea", "coffee", "orange juice"
    }

    # Convert user input to lowercase
    user_prompt_lower = user_prompt.lower()
    if any(keyword in user_prompt_lower for keyword in nutrition_keywords):
        answer_ = agent_ex.invoke({'input': user_prompt})
        #nutrition_answer = answer_.get('output', 'Sorry, I could not find nutrition information.')
        answer = fitness_chain.invoke({'question':answer_['output']})
    else:
        answer = fitness_chain.invoke({'question': user_prompt})

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
    st.write(answer)