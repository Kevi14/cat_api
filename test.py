import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Step 1: Define the problem and scope

# Step 2: Gather and preprocess data
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Cleaning and tokenization functions
def clean_text(text):
    # Remove special characters and punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    cleaned_text = cleaned_text.lower()
    
    # Remove extra whitespaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Apply any additional cleaning operations as per your requirements
    
    return cleaned_text

def tokenize_text(text):
    # Tokenize the text using NLTK's word_tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply any additional tokenization operations as per your requirements
    
    return tokens
# Preprocessing function for a single dialogue pair or context-response pair
def preprocess_dialogue_pair(context, response):
    # Clean the text
    cleaned_context = clean_text(context)
    cleaned_response = clean_text(response)

    # Tokenize the text
    context_tokens = tokenize_text(cleaned_context)
    response_tokens = tokenize_text(cleaned_response)

    return context_tokens, response_tokens

# Apply preprocessing to the entire dataset
preprocessed_data = []
for dialogue in dataset:
    context, response = dialogue['context'], dialogue['response']
    context_tokens, response_tokens = preprocess_dialogue_pair(context, response)
    preprocessed_data.append({'context': context_tokens, 'response': response_tokens})
# Step 3: Design the model architecture


# Step 4: Implement the model in PyTorch

class ConversationalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ConversationalModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Linear layer to map LSTM output to vocabulary space
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input):
        # Apply embedding
        embedded = self.embedding(input)
        
        # Pass through LSTM
        lstm_output, _ = self.lstm(embedded)
        
        # Flatten LSTM output
        flattened_output = lstm_output.contiguous().view(-1, lstm_output.size(2))
        
        # Apply linear layer to get logits
        logits = self.fc(flattened_output)
        
        return logits

# Step 5: Train the model
class ConversationalDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dialogue = self.data.iloc[index]
        context = dialogue['text'][0]
        response = dialogue['text'][1]

        # Apply any additional preprocessing to the context and response
        
        return context, response

# Step 6: Fine-tune the model (optional)

# Step 7: Create the conversational interface

# Step 8: Connect the model and interface

# Step 9: Test and iterate



def train_model(model, train_loader, val_loader, num_epochs, batch_size, learning_rate):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        running_loss = 0.0
        
        for batch_idx, (context, response) in enumerate(train_loader):
            # Move data to the device
            context = context.to(device)
            response = response.to(device)
            
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(context)
            
            # Calculate the loss
            loss = criterion(logits.view(-1, logits.size(-1)), response.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print batch loss every few batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0
        
        # Validate the model
        model.eval()  # Set the model to evaluation mode
        
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (context, response) in enumerate(val_loader):
                context = context.to(device)
                response = response.to(device)
                
                logits = model(context)
                
                loss = criterion(logits.view(-1, logits.size(-1)), response.view(-1))
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
    
    print("Training completed.")

def generate_response(model, input_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        # Preprocess the input text (e.g., clean and tokenize)
        preprocessed_input = preprocess_text(input_text)
        
        # Convert the preprocessed input to tensor
        input_tensor = torch.tensor(preprocessed_input).unsqueeze(0).to(device)
        
        # Forward pass through the model to generate output logits
        output_logits = model(input_tensor)
        
        # Get the predicted token indices (e.g., using argmax)
        predicted_indices = torch.argmax(output_logits, dim=-1).squeeze()
        
        # Convert the predicted token indices to tokens
        predicted_tokens = [index_to_token[index.item()] for index in predicted_indices]
        
        # Convert the tokens to a readable response
        response = " ".join(predicted_tokens)
        
        return response
# Entry point
if __name__ == '__main__':
    # Set up data, model, and training parameters
    # Instantiate the model
    model = ConversationalModel(vocab_size, embedding_dim, hidden_dim)

    # Prepare the dataset
    train_dataset = ConversationalDataset(train_data)
    val_dataset = ConversationalDataset(val_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, batch_size, learning_rate)

    # Generate responses using the trained model
    response = generate_response(model, input_text)

    # Use the model in your conversational interface
