"""
Factual Question-Answering Evaluation Module
=============================================
Evaluates factual knowledge retrieval of base model vs Memory MoE.
"""

import torch

FACTUAL_QA = [
    # Geography
    {"prompt": "What is the capital of France? Answer:", "answers": ["Paris"]},
    {"prompt": "What is the capital of Germany? Answer:", "answers": ["Berlin"]},
    {"prompt": "What is the capital of Japan? Answer:", "answers": ["Tokyo"]},
    {"prompt": "What is the capital of Italy? Answer:", "answers": ["Rome"]},
    {"prompt": "What is the capital of India? Answer:", "answers": ["New Delhi", "Delhi"]},
    {"prompt": "What is the capital of Spain? Answer:", "answers": ["Madrid"]},
    {"prompt": "What is the capital of Russia? Answer:", "answers": ["Moscow"]},
    {"prompt": "What is the capital of Canada? Answer:", "answers": ["Ottawa"]},
    {"prompt": "What is the largest ocean on Earth? Answer:", "answers": ["Pacific"]},
    {"prompt": "Which country is the largest by land area? Answer:", "answers": ["Russia"]},
    
    # Science & Space
    {"prompt": "What is the largest planet in our solar system? Answer:", "answers": ["Jupiter"]},
    {"prompt": "What planet is closest to the sun? Answer:", "answers": ["Mercury"]},
    {"prompt": "What planet is known as the Red Planet? Answer:", "answers": ["Mars"]},
    {"prompt": "What is the chemical symbol for gold? Answer:", "answers": ["Au"]},
    {"prompt": "What is the chemical symbol for water? Answer:", "answers": ["H2O"]},
    {"prompt": "What is the chemical symbol for helium? Answer:", "answers": ["He"]},
    {"prompt": "What force keeps planets in orbit? Answer:", "answers": ["gravity", "gravitation"]},
    {"prompt": "What is the speed of light in a vacuum? Answer:", "answers": ["299792458", "300,000", "300000"]},
    {"prompt": "How many bones are in the adult human body? Answer:", "answers": ["206"]},
    {"prompt": "What gas do plants absorb from the atmosphere? Answer:", "answers": ["carbon dioxide", "CO2"]},
    
    # Literature & History
    {"prompt": "Who wrote the play Hamlet? Answer:", "answers": ["Shakespeare", "William Shakespeare"]},
    {"prompt": "Who painted the Mona Lisa? Answer:", "answers": ["Da Vinci", "Leonardo da Vinci"]},
    {"prompt": "In which year did World War II end? Answer:", "answers": ["1945"]},
    {"prompt": "Who was the first President of the United States? Answer:", "answers": ["Washington", "George Washington"]},
    {"prompt": "Which empire built the Colosseum? Answer:", "answers": ["Roman", "Roman Empire"]},
    {"prompt": "Who discovered gravity when an apple fell? Answer:", "answers": ["Newton", "Isaac Newton"]},
    {"prompt": "Who was the first man to step on the Moon? Answer:", "answers": ["Armstrong", "Neil Armstrong"]},
    {"prompt": "Which country built the Great Pyramids of Giza? Answer:", "answers": ["Egypt"]},
    {"prompt": "What was the name of the ship that sank in 1912? Answer:", "answers": ["Titanic"]},
    {"prompt": "Who wrote the novel 1984? Answer:", "answers": ["George Orwell", "Orwell"]},
    
    # Math & Coding
    {"prompt": "What is the square root of 64? Answer:", "answers": ["8"]},
    {"prompt": "What is the value of 5 factorial? Answer:", "answers": ["120"]},
    {"prompt": "What is 15 multiplied by 6? Answer:", "answers": ["90"]},
    {"prompt": "What is the binary representation of the decimal number 5? Answer:", "answers": ["101"]},
    {"prompt": "What tag is used to create a link in HTML? Answer:", "answers": ["a", "anchor", "<a>"]},
    {"prompt": "Which programming language uses indentation to define code blocks? Answer:", "answers": ["Python"]},
    {"prompt": "What is the standard port number for HTTP? Answer:", "answers": ["80"]},
    {"prompt": "What does SQL stand for? Answer:", "answers": ["Structured Query Language"]},
    {"prompt": "In Python, which keyword is used to define a function? Answer:", "answers": ["def"]},
    {"prompt": "What is the time complexity of binary search? Answer:", "answers": ["O(log n)", "log n", "O(logn)"]},
]

@torch.no_grad()
def evaluate_factual_accuracy(model, tokenizer, device, num_tokens_to_generate=5):
    """
    Evaluate factual question-answering accuracy using greedy decoding.
    
    Returns:
        accuracy: float between 0.0 and 1.0
        results: list of tuples (prompt, generated, expected, is_correct)
    """
    was_training = model.training
    model.eval()
    
    # Ensure base model is also in eval mode
    if hasattr(model, "base_model"):
        model.base_model.eval()
        
    correct = 0
    results = []
    
    for qa in FACTUAL_QA:
        prompt = qa["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Greedy generation
        generated_ids = inputs["input_ids"]
        for _ in range(num_tokens_to_generate):
            # Pass past_key_values = None to evaluate clean generation
            if hasattr(model, "base_model"):
                outputs = model(input_ids=generated_ids)
            else:
                # Evaluating base model directly
                outputs = model(input_ids=generated_ids)
                
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        generated_text = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Check if expected answers match (case-insensitive substring match)
        is_correct = False
        for expected in qa["answers"]:
            if expected.lower() in generated_text.lower():
                is_correct = True
                break
                
        if is_correct:
            correct += 1
            
        results.append((prompt, generated_text, qa["answers"], is_correct))
        
    accuracy = correct / len(FACTUAL_QA)
    
    # Restore model training mode
    if was_training:
        model.train()
        if hasattr(model, "base_model"):
            model.base_model.eval() # Base always remains frozen eval
            
    return accuracy, results
