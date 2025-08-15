import requests, time, json, pickle, re, os

def chat(text):
    url = "http://180.93.182.28:11435/api/generate"
    prompt = f"""{text}"""
    
    payload = {
        "model": "qwen3:14b",
        "prompt": prompt,
        "stream": False,
        "think": True
    }

    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}", response.text)
        return None
    
def chat2(text):
    start = time.time()
    url = "https://openrouter.ai/api/v1/chat/completions"   

    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
    }

    prompt = f"""{text}"""
    models = ["moonshotai/kimi-k2:free", 
              "deepseek/deepseek-chat-v3-0324:free", 
              "deepseek/deepseek-r1-0528:free",
              "tngtech/deepseek-r1t2-chimera:free",
              "qwen/qwen3-235b-a22b:free"]
    data = {
        "model": models[3],  
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,  # Moderate creativity
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        # Always check status
        if response.status_code != 200:
            print(f"HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return "vl"
        
        result = response.json()
        message_content = result['choices'][0]['message']['content']
        end = time.time()

        print(f"running time {end - start:.2f}")

        return message_content

    except Exception as e:
        print(f"Exception occurred: {e}")
    
# ans = chat2("""Generate 10 texts (each must be 1000 words long) that are directly related to SDG 1 (No Poverty) without explicitly mentioning it. Each text should also include some noise—such as unrelated details, minor references to other SDGs (e.g., health, education, climate, gender, water, life, justice), or slightly off-topic elements.

# Ensure diversity in scenarios and keep the language natural. Avoid direct terms like 'SDG 1,' 'poverty reduction,' or 'extreme poor.""")

prompt = """Generate exactly 10 texts that directly address one of these themes: Poverty Eradication, Social Protection Systems, Economic Empowerment, Resilience to Shocks & Disasters, Bridging the wealth gap, Affordable housing and sanitation, Government accountability in poverty programs—without explicitly naming the goal. Each text should directly relate to SDG 1 - No poverty. Each text should adopt a formal academic writing style, resembling abstract sections from peer-reviewed journal articles such as literature reviews, methodology discussions, or results analyses. REMEMBER, Each text must be long, about 40 sentences.
Ensure all texts are ready for dataset use to fine tune model. Format each text like this:  

### Text 1 ###  
[Your first text here.]  

### Text 2 ###  
[Your second text here.] 
            
### Text 3 ###  
[Your third text here.] 
            
...  
            
### Text 10 ###  
[Your tenth text here.] 
"""

prompt = """Generate exactly 5 texts that NOT address one of these themes: Poverty Eradication, Social Protection Systems, Economic Empowerment, Resilience to Shocks & Disasters, Bridging the wealth gap, Affordable housing and sanitation, Government accountability in poverty programs—without explicitly naming the goal. Each text MUST not relate to SDG 1 - No poverty. Each text should adopt a formal academic writing style, resembling abstract sections from peer-reviewed journal articles such as literature reviews, methodology discussions, or results analyses. REMEMBER, Each text must be long, about 30 sentences.
Ensure all texts are ready for dataset use to fine tune model. Format each text like this:  

### Text 1 ###  
[Your first text here.]  

### Text 2 ###  
[Your second text here.] 
            
### Text 3 ###  
[Your third text here.] 
            
### Text 4 ###  
[Your fourth text here.] 
            
### Text 5 ###  
[Your fifth text here.] """

def main(filename):
    matches = []
    if os.path.exists(filename):
        with open(filename, "rb") as file:
            matches = pickle.load(file)
    for i in range(10):
        ans = chat2(prompt)
        while(ans == "vl"):
            print("sleep for 3 seconds")
            time.sleep(3)
            ans = chat2(prompt)
            
        pattern = r'### .*? ###(.*?)(?=\s*### |\Z)'
        matches.extend(re.findall(pattern, ans, flags=re.DOTALL))

        if (i % 3 == 0):
            with open(filename, "wb") as file:
                pickle.dump(matches, file)
    with open(filename, "wb") as file:
                pickle.dump(matches, file)

main("output14.pkl")