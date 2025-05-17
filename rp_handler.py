from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import runpod

# Load the model and tokenizer from Hugging Face Hub
model = AutoModelForSeq2SeqLM.from_pretrained("JMwagunda/ENG-GIR-MODEL")
tokenizer = AutoTokenizer.from_pretrained("JMwagunda/ENG-GIR-MODEL")

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Model and tokenizer loaded from Hugging Face Hub.")

def translate(text, source_lang="sw", target_lang="en", max_length=128, num_beams=5, num_return_sequences=1):
    # Language tokens
    lang_token = {
        'en': '<en>',
        'sw': '<sw>'
    }

    # Add language token to the input text
    input_text = f"{lang_token[source_lang]} {text}"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = input_ids.to(device)

    # Generate translation
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True  # Stop generation when the model outputs the EOS token
    )

    # Decode the generated tokens to text
    translated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return translated_texts

def handler(event):
    try:
        # Check if the input contains the required text
        if "input" not in event or "text" not in event["input"]:
            return {"error": "No text provided for translation"}
        
        # Get parameters from the request
        text = event["input"]["text"]
        source_lang = event["input"].get("source_lang", "sw")
        target_lang = event["input"].get("target_lang", "en")
        max_length = event["input"].get("max_length", 128)
        num_beams = event["input"].get("num_beams", 5)
        
        # Validate languages
        valid_langs = ["en", "sw"]
        if source_lang not in valid_langs or target_lang not in valid_langs:
            return {"error": f"Invalid language. Supported languages are: {', '.join(valid_langs)}"}
        
        # Perform translation
        translated_texts = translate(
            text, 
            source_lang=source_lang, 
            target_lang=target_lang,
            max_length=max_length,
            num_beams=num_beams,
        )
        
        # Return the translated text
        return {
            "success": True,
            "translations": translated_texts,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False, 
            "error": str(e), 
            "traceback": traceback.format_exc()
        }


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})