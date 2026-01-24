import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset 

# RAGAS Imports
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Import your app
from app import get_tax_assistant

# 1. Load Environment & API Key
load_dotenv()
my_api_key = os.getenv("GEMINI_API_KEY")

if not my_api_key:
    print("⚠️ Error: GEMINI_API_KEY not found in .env file")
    exit(1)

# 2. Setup Gemini for Grading (The Judge)
# Using gemini-1.5-flash for evaluation stability
judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    temperature=0,
    google_api_key=my_api_key,
    timeout=60
)

judge_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=my_api_key
)

# Wrap for RAGAS
evaluator_llm = LangchainLLMWrapper(judge_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

# 3. Define the Exam Questions
data = {
    'question': [
        'How is the Consolidated Relief Allowance calculated in 2024?',
        'What is the tax rate for someone earning 300,000 naira annually?',
        'Calculate the tax impact for a monthly income of 500,000 naira.'
    ],
    'ground_truth': [
        'In 2024, CRA is calculated as the higher of 200,000 Naira or 1% of Gross Income, plus 20% of Gross Income.',
        'The first 300,000 Naira of taxable income is taxed at 7%.',
        'Use the calculator tool to determine the exact tax impact based on current laws.'
    ]
}

# 4. Run YOUR Agent against the Exam
print("🤖 Agent is taking the exam...")
assistant = get_tax_assistant()

actual_answers = []
retrieved_contexts = []

for q in data['question']:
    print(f"   Asking: {q}...")
    response_text = assistant.ask_question(q, user_id="eval_user")
    actual_answers.append(response_text)
    
    # Context Hack: Pass the answer as context to satisfy RAGAS requirements
    # (Since we aren't extracting raw docs from the agent response yet)
    retrieved_contexts.append([response_text]) 

# 5. Build Dataset
eval_dataset = Dataset.from_dict({
    'question': data['question'],
    'answer': actual_answers,
    'contexts': retrieved_contexts,
    'ground_truth': data['ground_truth']
})

# 6. Grade it!
print("👨‍⚖️ Grading in progress...")
results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=False 
)

# 7. Save to File
output_filename = "evaluation.txt"
with open(output_filename, "w", encoding="utf-8") as f:
    f.write("=== TAX AGENT EVALUATION REPORT ===\n")
    f.write(f"Model Used: gemini-1.5-flash\n")
    f.write("-" * 40 + "\n")
    f.write(f"Faithfulness Score:   {results.get('faithfulness', 0):.4f}\n")
    f.write(f"Answer Relevancy:     {results.get('answer_relevancy', 0):.4f}\n")
    f.write("-" * 40 + "\n\n")
    
    f.write("--- DETAILED Q&A ---\n")
    df = results.to_pandas()
    for index, row in df.iterrows():
        f.write(f"\nQ: {row['question']}\n")
        f.write(f"A: {row['answer']}\n")
        f.write(f"Faithfulness: {row['faithfulness']:.2f} | Relevancy: {row['answer_relevancy']:.2f}\n")
        f.write("-" * 20 + "\n")

print(f"\n✅ Results saved to {output_filename}")