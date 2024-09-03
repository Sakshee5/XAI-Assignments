import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def load_model_and_tokenizer(model_name, device="cuda:0"):
    """
    Load the GPT-Neo 1.3B model and tokenizer with mixed precision.
    """
    # Specifies the data type for model parameters and moves the model to the specified device (e.g., GPU for faster processing) in evaluation mode
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_embedding_matrix(model):
    """
    Retrieve the embedding matrix from the model.
    """
    return model.transformer.wte.weight


def generate(model, input_embeddings, num_tokens=50):
    """
    Generate text from the given input embeddings.

    Args:
        model: The language model to use for generation.
        input_embeddings (torch.Tensor): The input embeddings to generate text from.
        num_tokens (int): The number of tokens to generate.

    Returns:
        list: The generated tokens as a list of integers.
    """
    model.eval()
    embedding_matrix = get_embedding_matrix(model)
    input_embeddings = input_embeddings.clone()

    generated_tokens = []
    with torch.no_grad():
        for _ in range(num_tokens):
            outputs = model(inputs_embeds=input_embeddings)
            next_token_logits = outputs.logits[:, -1, :]          # The model’s predictions for the next token in the sequence.
            next_token = torch.argmax(next_token_logits, dim=-1)  # Chooses the token with the highest probability as the next token.
            generated_tokens.append(next_token.item())
            
            next_token_embedding = embedding_matrix[next_token].unsqueeze(0)
            input_embeddings = torch.cat((input_embeddings, next_token_embedding), dim=1) # Updated with the newly generated token embedding.
    
    return generated_tokens

def calc_loss(model, embeddings, embeddings_attack, embeddings_target, targets):
    """
    Calculate the loss for the adversarial attack.

    Args:
        model: The language model.
        embeddings (torch.Tensor): The initial prompt embeddings.
        embeddings_attack (torch.Tensor): The adversarial embeddings.
        embeddings_target (torch.Tensor): The target embeddings.
        targets (torch.Tensor): The target tokens.

    Returns:
        tuple: Loss and logits for the adversarial attack.
    """

    # Combines different sets of embeddings (original, attack, target) to feed into the model.
    # By including the target embeddings, the model can use the context provided by the initial and adversarial embeddings to make predictions.
    full_embeddings = torch.cat([embeddings, embeddings_attack, embeddings_target], dim=1)
    outputs = model(inputs_embeds=full_embeddings)
    logits = outputs.logits         # model predictions
    loss_start = embeddings.shape[1] + embeddings_attack.shape[1] - 1
    loss = nn.CrossEntropyLoss()(logits[:, loss_start:-1, :].reshape(-1, logits.size(-1)), targets)
    return loss, logits


def create_one_hot_and_embeddings(tokens, embed_weights, device):
    """
    Create one-hot encoded vectors and their corresponding embeddings.

    Args:
        tokens (torch.Tensor): The input tokens.
        embed_weights (torch.Tensor): The embedding matrix.
        model: The language model.

    Returns:
        tuple: One-hot encoded vectors and their embeddings.
    """
    one_hot = torch.zeros((len(tokens), embed_weights.size(0)), device=device)
    one_hot[range(len(tokens)), tokens] = 1.0
    embeddings = one_hot @ embed_weights  # produces embeddings from the one-hot vectors using the embedding matrix.
    embeddings = embeddings.unsqueeze(0)  # adds a batch dimension at position 0
    return one_hot, embeddings

def run_attack(
    model,
    tokenizer,
    fixed_prompt,
    control_prompt_init,
    target_text,
    num_steps=100,
    step_size=0.01,
    device="cuda"
):
    """
    Run adversarial attack to induce the model to generate target text.
    """
    embed_weights = get_embedding_matrix(model)
    
    # Tokenize inputs; return_tensors="pt" returns the tokens as a PyTorch tensor instead of a list
    fixed_tokens = tokenizer.encode(fixed_prompt, return_tensors="pt").to(device)
    control_tokens = tokenizer.encode(control_prompt_init, return_tensors="pt").to(device)
    target_tokens = tokenizer.encode(target_text, return_tensors="pt").to(device)
    
    # Create embeddings
    _, fixed_embeddings = create_one_hot_and_embeddings(fixed_tokens[0], embed_weights, device)
    _, control_embeddings = create_one_hot_and_embeddings(control_tokens[0], embed_weights, device)
    target_one_hot, target_embeddings = create_one_hot_and_embeddings(target_tokens[0], embed_weights, device)
    
    # Initialize adversarial perturbation with a small random noise, which can help explore the embedding space more effectively
    adv_perturbation = torch.randn_like(control_embeddings, device=device).detach()*0.01
    adv_perturbation.requires_grad = True

    optimizer = torch.optim.Adam([adv_perturbation], lr=step_size)
    results = []

    for step in range(num_steps):
        optimizer.zero_grad()
        loss, logits = calc_loss(
            model,
            fixed_embeddings,
            control_embeddings + adv_perturbation,
            target_embeddings,
            target_tokens[0]
        )
        loss.backward(retain_graph=True)
        
        # Apply gradient clipping to prevent gradients from exploding
        torch.nn.utils.clip_grad_norm_([adv_perturbation], max_norm=1.0)
        
        optimizer.step()

        if step % 10 == 0 or step == num_steps - 1:
            print(f"\nStep {step}/{num_steps} - Loss: {loss.item():.4f}")

            full_embeddings = torch.cat([fixed_embeddings, control_embeddings + adv_perturbation], dim=1)
            generated_tokens = generate(model, full_embeddings, num_tokens=50)
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"\n*** Generated Text at Step {step} ***")
            print(generated_text)

            # Save the step number, loss, and generated text to a list
            results.append({
                "Step": step,
                "Loss": loss.item(),
                "Generated Text": generated_text
            })

    # Convert the list of results into a DataFrame and save as an Excel file
    df = pd.DataFrame(results)
    df.to_excel("attack_results.xlsx", index=False)


fixed_prompt = "As she turned the corner, she noticed"
control_prompt_init = "!!!"
target_text = "As she turned the corner, she noticed a shadowy figure standing by her car, holding a knife."

model_name = "EleutherAI/gpt-neo-1.3B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model_and_tokenizer(model_name=model_name, device=device)

run_attack(
    model,
    tokenizer,
    fixed_prompt,
    control_prompt_init,
    target_text,
    device=device
)

