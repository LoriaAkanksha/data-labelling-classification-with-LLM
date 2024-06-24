from dotenv import load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
import os
from langchain.schema import HumanMessage
from warnings import filterwarnings
import pandas as pd
filterwarnings("ignore")

load_dotenv()

categories = ["IT", "Product", "Marketing", "Sales", "Admin", "Editorial", "CXO"]

# Load job titles from CSV file
job_titles_df = pd.read_csv("/home/akanksha/Downloads/Unique Job Titles Hubspot - Job Title.csv")
#job_titles_df=job_titles_df.head(1000)

# Extract job titles from the DataFrame
job_titles = job_titles_df["Job_titles"].tolist()

# Get deployment name from environment variable
llm_deployment_name = os.getenv("llm_deployement_name")

# Create a message for classification
instructions = HumanMessage(
    content=f"Classify the job titles into one of the categories: {categories}\nCategory:"
)

# Initialize AzureChatOpenAI model
turbo_llm = AzureChatOpenAI(
    deployment_name=llm_deployment_name,
    model_name="gpt-35-turbo-16k"
)

# Create an empty list to store the results
classified_jobs = []

# Classify each job title
for job_title in job_titles:
    instructions.content = f"Classify the job title '{job_title}' into one of the categories: {categories}\nCategory:"
    result = turbo_llm([instructions])
    classified_jobs.append({"Job Title": job_title, "Category": result})

# Convert the list of dictionaries into a DataFrame
classified_jobs_df = pd.DataFrame(classified_jobs)

# Write the DataFrame to a CSV file
classified_jobs_df.to_csv("/home/akanksha/Downloads/classified_jobs.csv", index=False)

print("Classification results saved to classified_jobs.csv")
