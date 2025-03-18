import ollama

def diagnose_ulcer(symptoms):
    response = ollama.chat(
        model='mistral',
        messages=[
            {"role": "system", "content": "You are a medical assistant diagnosing stomach ulcers."},
            {"role": "user", "content": f"My symptoms are: {symptoms}"}
        ]
    )
    return response['message']['content']

symptoms = input("Enter your symptoms: ")
diagnosis = diagnose_ulcer(symptoms)
print("Diagnosis:", diagnosis)
