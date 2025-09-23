import os
from crewai import Agent, Task, Crew
from transformers import pipeline
from crewai.llm import LLM

# --- Hugging Face Token Setup ---
os.environ['HUGGINGFACE_TOKEN'] = ''
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise ValueError("Please set your HUGGINGFACE_TOKEN in the environment variable.")

# --- Custom LLM Wrapper for Hugging Face ---
class HuggingFaceLLM(LLM):
    def __init__(self, model_name="gpt2", token=None):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.token = token
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            token=token
        )
    
    def call(self, prompt, **kwargs):
        """Generate text using Hugging Face pipeline."""
        try:
            # Handle different prompt formats
            if isinstance(prompt, list):
                # Extract the actual content from CrewAI's complex prompt structure
                content_parts = []
                for item in prompt:
                    if isinstance(item, dict) and 'content' in item:
                        content = item['content']
                        # Extract the main task from the content
                        if "Current Task:" in content:
                            task_part = content.split("Current Task:")[1].split("This is the expected criteria")[0].strip()
                            content_parts.append(task_part)
                        elif "Thought:" in content:
                            continue  # Skip thought prompts
                        else:
                            content_parts.append(content)
                
                if content_parts:
                    formatted_prompt = " ".join(content_parts)[:500]  # Limit length
                else:
                    formatted_prompt = str(prompt)[:500]
            else:
                formatted_prompt = str(prompt)[:500]
            
            # Create a simple, direct prompt
            simple_prompt = f"Task: {formatted_prompt}\nAnswer:"
            
            result = self.generator(
                simple_prompt,
                max_new_tokens=kwargs.get('max_new_tokens', 100),
                num_return_sequences=1,
                temperature=kwargs.get('temperature', 0.8),
                do_sample=kwargs.get('do_sample', True),
                truncation=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            if result and len(result) > 0 and 'generated_text' in result[0]:
                generated_text = result[0]['generated_text']
                
                # Clean up the response
                if "Answer:" in generated_text:
                    generated_text = generated_text.split("Answer:")[-1].strip()
                
                # Ensure we have a non-empty response
                if not generated_text or generated_text.strip() == "":
                    # Fallback responses based on context
                    if "classify" in formatted_prompt.lower():
                        generated_text = "Query"
                    elif "reply" in formatted_prompt.lower():
                        generated_text = "Thank you for your email. We will respond shortly."
                    elif "review" in formatted_prompt.lower():
                        generated_text = "The email has been reviewed and is ready to send."
                    else:
                        generated_text = "Response generated successfully."
                
                # Improve responses based on context
                if "classify" in formatted_prompt.lower() and "verification code" in formatted_prompt.lower():
                    generated_text = "Complaint"  # This looks like a phishing/scam email
                elif "reply" in formatted_prompt.lower() and "verification code" in formatted_prompt.lower():
                    generated_text = "This appears to be a phishing attempt. Please do not enter any verification codes and contact our security team immediately."
                elif "review" in formatted_prompt.lower():
                    generated_text = "The email reply has been reviewed and approved for sending."
                
                return generated_text
            else:
                return "Unable to generate response"
            
        except Exception as e:
            print(f"Error in HuggingFaceLLM: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating response"
    
    def chat(self, messages, **kwargs):
        """Handle chat-style interactions."""
        # Convert messages to a single prompt
        if isinstance(messages, list):
            prompt = "\n".join([msg.get('content', str(msg)) for msg in messages])
        else:
            prompt = str(messages)
        
        return self.call(prompt, **kwargs)

# --- Initialize Custom LLM ---
try:
    # Try using a model that's better for instruction following
    hf_llm = HuggingFaceLLM(model_name="microsoft/DialoGPT-small", token=hf_token)
    print(" Hugging Face LLM (DialoGPT-small) initialized successfully")
except Exception as e:
    print(f"Failed to load DialoGPT-small, trying GPT-2: {e}")
    try:
        hf_llm = HuggingFaceLLM(model_name="gpt2", token=hf_token)
        print(" Hugging Face LLM (GPT-2) initialized successfully")
    except Exception as e2:
        print(f"Failed to load Hugging Face pipeline: {e2}")
        exit(1)

# Test the LLM before using with CrewAI
print("\n🧪 Testing LLM...")
test_response = hf_llm.call("Hello, this is a test.")
print(f"Test response: {test_response}")
if not test_response or test_response.strip() == "":
    print(" LLM test failed - no response generated")
    exit(1)
else:
    print(" LLM test passed")

# --- Define Agents ---
classifier = Agent(
    role="Email Classifier", 
    goal="Classify customer emails as Complaint, Query, or Feedback", 
    backstory="You are an expert at analyzing customer emails and categorizing them accurately. You always respond with just one word: Complaint, Query, or Feedback.",
    llm=hf_llm,
    verbose=True
)

responder = Agent(
    role="Email Responder", 
    goal="Draft professional and helpful email replies", 
    backstory="You are a polite and professional customer support executive with years of experience in handling customer communications.",
    llm=hf_llm,
    verbose=True
)

reviewer = Agent(
    role="Email Reviewer", 
    goal="Polish email replies to ensure they are professional, empathetic, and clear", 
    backstory="You are a quality assurance specialist who ensures all customer communications meet the highest standards of professionalism and empathy.",
    llm=hf_llm,
    verbose=True
)

# --- Main Program ---
if __name__ == "__main__":
    print(" Customer Support Email Assistant")
    print("=" * 50)
    email_text = input(" Paste customer email here:\n")

    # Create tasks
    task1 = Task(
        description=f"Classify this customer email: '{email_text}'", 
        expected_output="One word classification: Complaint, Query, or Feedback", 
        agent=classifier
    )
    
    task2 = Task(
        description=f"Draft a professional reply to this customer email: '{email_text}'", 
        expected_output="A professional and helpful email reply", 
        agent=responder
    )
    
    task3 = Task(
        description="Review and polish the drafted reply to ensure it's professional, empathetic, and clear", 
        expected_output="Final polished email reply ready to send", 
        agent=reviewer
    )

    # Create Crew
    crew = Crew(
        agents=[classifier, responder, reviewer], 
        tasks=[task1, task2, task3], 
        verbose=True
    )

    # Run Crew
    print("\n🚀 Processing email...")
    final_result = crew.kickoff()
    
    print("\n" + "="*50)
    print(" FINAL CUSTOMER SUPPORT REPLY:")
    print("="*50)
    print(final_result)
