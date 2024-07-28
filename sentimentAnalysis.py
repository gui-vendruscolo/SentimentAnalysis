import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('imdb')

train_data = dataset['train']
test_data = dataset['test']

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, pin_memory=True)

batch = next(iter(train_loader))
print(batch['input_ids'].shape)
print(batch['label'].shape)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, device, acumulation_steps=8):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in tqdm(iterator):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()

        if len(iterator) % acumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_acc += (logits.argmax(dim=1) == labels).sum().item() / len(labels)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in tqdm(iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            logits = output.logits

            epoch_loss += loss.item()
            epoch_acc += (logits.argmax(dim=1) == labels).sum().item() / len(labels)

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

n_epochs = 3

for epoch in range(n_epochs):
    print(f"Now training. Epoch number {epoch}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    
    print(f"Now validating. Epoch number {epoch}")
    valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)

    print(f'Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train accuracy: {train_acc*100:.2f}%')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid accuracy: {valid_acc*100:.2f}%')
