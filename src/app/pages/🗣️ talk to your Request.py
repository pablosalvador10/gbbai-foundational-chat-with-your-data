import streamlit as st
from src.utilsfunc import save_uploaded_file
import dotenv
import os
from utils.ml_logging import get_logger
from src.ocr.cosmosDB_indexer import CosmosDBIndexer
from src.app.prompts import get_chat_cosmos_db_prompt, get_cosmos_db_prompt

# Load environment variables
dotenv.load_dotenv(".env")

# Set up logger
logger = get_logger()

from src.ocr.transformer import GPT4VisionManager
from src.aoai.azure_openai import AzureOpenAIManager

# Initialize GPT4VisionManager in session state
if "gpt4_vision_manager" not in st.session_state:
    st.session_state.gpt4_vision_manager = GPT4VisionManager()

# Initialize GPT4VisionManager in session state
if "gpt4_manager" not in st.session_state:
    st.session_state.gpt4_manager = AzureOpenAIManager(
        api_key=os.getenv("AZURE_AOAI_KEY"),
        api_version=os.getenv("AZURE_AOAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_AOAI_API_ENDPOINT"),
        completion_model_name=os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"))

if "cosmos_manager" not in st.session_state:
    st.session_state.cosmos_client = CosmosDBIndexer(database_name="gbbai-qualifiction-db",
                                                     container_name="gbbai-qualifiction")

st.title("🤖 AI RequestGPT")

# About App expander
with st.expander("About this App"):
    st.markdown(
        """
        ### 🌟 Application Overview
        This application demonstrates the power of Azure OpenAI in a real-time interaction context. It seamlessly integrates Azure OpenAI services to provide a sophisticated request interaction experience.

        #### Key Features:
        - **Request Interaction**: Utilizes Azure OpenAI to understand and interact with your request data in real-time.
        - **Decision Making**: Employs Azure OpenAI GPT-4 for advanced decision-making based on your request data.
        - **Real-Time Updates**: Leverages Azure OpenAI to provide real-time insights and updates about your requests.
        """,
        unsafe_allow_html=True,
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image_paths = []
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    image_paths.append(file_path)

# React to user input
if prompt := st.chat_input("How can I assist you today?"):
    with st.spinner('Generating response...'):
        COSMOS_DB_PROMPT_WITH_PROMPT = get_cosmos_db_prompt(prompt)
        response_query = st.session_state.gpt4_vision_manager.call_gpt4v_image(
            image_file_paths=image_paths,
            user_instruction=COSMOS_DB_PROMPT_WITH_PROMPT,
            system_instruction='''You are and Cosmos DB engineer expert tasked with translating natural
                language queries related to project requests into specific Cosmos DB queries. The database schema
                includes fields like `RequestTitle`, `Requester`, `RequesterEmail`, `Partner`, `ProjectedWorkHours`,
                `ExpectedStartDate`, `MSXID`, `TPID`, `PrimarySolutionArea`, `SecondarySolutionArea`, `CustomerName`, 
                `OperatingUnit`, `ProblemDescription`, `ProjectedACR`, `NecessarySkills`, `AzureAIServices`, `EngagementCountry`, 
                `EngagementRegion`, `MonthlyUsage`, `Attachment`, `CreatedDate`, `RequestId`, `Status`, `AssignedTo`, `AssignedDate`, 
                `Approved`, `ApprovedDate`, and `ApprovedBy`.''',
            ocr=True,
            use_vision_api=True,
            display_image=False,
            temperature=0.7,
            max_tokens=450,
            top_p=1.0
        )

        logger.info(f"Generated CosmosDb query: {response_query}")

        data = st.session_state.cosmos_client.execute_query(response_query)

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare conversation history for the AI model
        conversation_history = [
            message for message in st.session_state["messages"]
            if message["role"] != "system"
        ]

        CHAT_COSMOS_DB_PROMPT_WITH_PROMPT = get_chat_cosmos_db_prompt(json_response=data, prompt=prompt)
    
        response = st.session_state.gpt4_vision_manager.call_gpt4v_image(
            image_file_paths=image_paths,
            system_instruction="You are an AI assistant that helps people find information. Please be precise, polite, and concise.",
            user_instruction=CHAT_COSMOS_DB_PROMPT_WITH_PROMPT,
            ocr=True,
            use_vision_api=True,
            display_image=False,
            max_tokens=2000,
            seed=42,
        )

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})