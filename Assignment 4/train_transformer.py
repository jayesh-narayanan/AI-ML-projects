import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

class Vocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<>'  
        self.EOS_TOKEN = '<EOS>'  
        self.UNK_TOKEN = '<UNK>'

        # Add special tokens
        self.add_token(self.PAD_TOKEN)
        self.add_token(self.SOS_TOKEN)
        self.add_token(self.EOS_TOKEN)
        self.add_token(self.UNK_TOKEN)

    def add_token(self, token):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build_from_sequences(self, sequences):
        for seq in sequences:
            for token in seq:
                self.add_token(token)

    def encode(self, tokens):
        return [self.token2idx.get(t, self.token2idx[self.UNK_TOKEN]) for t in tokens]

    def decode(self, indices):
        return [self.idx2token[i] for i in indices if i in self.idx2token]

    def __len__(self):
        return len(self.token2idx)
    
class MazeDataset(Dataset):
    def __init__(self, csv_path, vocab=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train

        # Parse sequences
        self.inputs = [ast.literal_eval(s) for s in self.df['input_sequence']]
        self.outputs = [ast.literal_eval(s) for s in self.df['output_path']]

        # Build or use vocabulary
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_from_sequences(self.inputs + self.outputs)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        output_seq = self.outputs[idx]

        # Encode sequences
        input_encoded = self.vocab.encode(input_seq)
        output_encoded = self.vocab.encode(output_seq)

        return {
            'input': torch.tensor(input_encoded, dtype=torch.long),
            'output': torch.tensor(output_encoded, dtype=torch.long),
            'input_len': len(input_encoded),
            'output_len': len(output_encoded)
        }

def collate_fn(batch):
    """Pad sequences to max length in batch"""
    inputs = [item['input'] for item in batch]
    outputs = [item['output'] for item in batch]
    input_lens = [item['input_len'] for item in batch]
    output_lens = [item['output_len'] for item in batch]

    # Pad sequences
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    outputs_padded = nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=0)

    return {
        'input': inputs_padded,
        'output': outputs_padded,
        'input_len': torch.tensor(input_lens),
        'output_len': torch.tensor(output_lens)
    }

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as per 'Attention is All You Need'"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create sinusoidal positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class TransformerMazeSolver(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embedding layer for encoder and decoder 
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Scale embeddings by sqrt(d_model) as mentioned in the paper
        nn.init.normal_(self.embedding.weight, mean=0, std=d_model**-0.5)
        with torch.no_grad():
            self.embedding.weight[0] = 0  # Zero padding token

        # Sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder stack with self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder stack with masked self-attention and cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask for decoder (as per section 3.2.3 of the paper)
        Returns: FloatTensor where 0.0 means 'attend' and -inf means 'mask'
        """
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq, pad_idx=0):
        """Create padding mask (True for padding positions)"""
        return (seq == pad_idx)

    def forward(self, src, tgt):
        """
        Forward pass as described in the Transformer paper
        Args:
            src: [batch, src_seq_len] - input sequence (maze description)
            tgt: [batch, tgt_seq_len] - target sequence (shifted right for teacher forcing)
        Returns:
            [batch, tgt_seq_len, vocab_size]
        """
        # Create masks
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)

        # Causal mask for decoder (masked self-attention)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Encoder: self-attention on input sequence
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

        # Decoder: masked self-attention + cross-attention to encoder output
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )

        # Project to vocabulary
        output = self.fc_out(output)
        return output

    def predict(self, src, max_len=50, start_token=1, end_token=2):
        """Generate predictions autoregressively with causal masking"""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)

            # Encoder self attention
            src_padding_mask = self.create_padding_mask(src)
            src_emb = self.embedding(src) * math.sqrt(self.d_model)
            src_emb = self.pos_encoder(src_emb)
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)

            # Initialize decoder with start token
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=self.device)
            predictions = []

            for i in range(max_len):
                # Create mask for current sequence length
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

                # Embed and positionally encode target
                tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
                tgt_emb = self.pos_encoder(tgt_emb)

                # Decoder: masked self-attention + cross-attention
                output = self.transformer_decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                # Get next token prediction (only last position)
                output = self.fc_out(output)
                next_token = output[:, -1, :].argmax(dim=-1)
                predictions.append(next_token)

                # Check if all sequences have generated end token
                if (next_token == end_token).all():
                    break

                # Append prediction to target sequence for next iteration
                tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)

            # Stack predictions
            if predictions:
                predictions = torch.stack(predictions, dim=1)
            else:
                predictions = torch.zeros(batch_size, 0, dtype=torch.long).to(self.device)

            return predictions
        
def train_epoch(model, dataloader, criterion, optimizer, device, vocab):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        src = batch['input'].to(device)
        tgt = batch['output'].to(device)

        # 1. Prepare Decoder Input (Shift Right)
        sos_idx = vocab.token2idx.get('<PATH_START>',
                  vocab.token2idx.get('<SOS>',
                  vocab.token2idx.get('<>', 1)))

        # [SOS, t1, t2, ..., tn-1]
        tgt_input = torch.cat([
            torch.full((tgt.size(0), 1), sos_idx, dtype=torch.long).to(device),
            tgt[:, :-1]
        ], dim=1)

        optimizer.zero_grad()

        # 2. Forward Pass
        output = model(src, tgt_input)

        # 3. Calculate Loss
        output_flat = output.reshape(-1, output.size(-1))
        tgt_flat = tgt.reshape(-1)
        loss = criterion(output_flat, tgt_flat)

        # 4. Backward Pass
        loss.backward()

        # 5. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 6. Step
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, vocab):
    """Evaluate the model and return loss"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            src = batch['input'].to(device)
            tgt = batch['output'].to(device)

            sos_token = vocab.token2idx.get('<PATH_START>',
                        vocab.token2idx.get('<>',
                        vocab.token2idx.get('<SOS>', 1)))
            tgt_input = torch.cat([
                torch.full((tgt.size(0), 1), sos_token, dtype=torch.long).to(device),
                tgt[:, :-1]
            ], dim=1)

            # Forward pass
            output = model(src, tgt_input)

            # Calculate loss
            output_flat = output.reshape(-1, output.size(-1))
            tgt_flat = tgt.reshape(-1)
            loss = criterion(output_flat, tgt_flat)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def calculate_accuracies(model, dataloader, vocab, device, start_token, end_token):
    """
    Calculate both token-level and sequence-level accuracies

    Returns:
        token_accuracy: Percentage of correctly predicted tokens
        sequence_accuracy: Percentage of perfectly matched sequences
    """
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Calculating Accuracy', leave=False):
            src = batch['input'].to(device)
            tgt = batch['output'].to(device)

            # Generate predictions
            predictions = model.predict(src, max_len=50, start_token=start_token, end_token=end_token)

            # Compare with ground truth
            for i in range(src.size(0)):
                pred = predictions[i].cpu().tolist()
                true = tgt[i].cpu().tolist()

                # Remove padding from ground truth
                true = [t for t in true if t != 0]

                # Truncate at end token if present
                if end_token in pred:
                    pred = pred[:pred.index(end_token) + 1]

                # Sequence-level accuracy (exact match)
                if pred == true:
                    correct_sequences += 1
                total_sequences += 1

                # Token-level accuracy
                min_len = min(len(pred), len(true))
                for j in range(min_len):
                    if pred[j] == true[j]:
                        correct_tokens += 1
                    total_tokens += 1

                # Count extra/missing tokens as incorrect
                total_tokens += abs(len(pred) - len(true))

    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    sequence_accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0

    return token_accuracy, sequence_accuracy

def train_with_checkpoints(model, train_loader, val_loader, test_loader,
                                   criterion, optimizer, vocab, device,
                                   num_epochs, checkpoint_dir='checkpoints'):
    """
    Train model starting with existing metrics (from lost checkpoints)

    Args:
        existing_metrics: Dictionary with metrics from previous epochs (optional)
        All other args same as before
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get special tokens
    start_token = vocab.token2idx.get('<PATH_START>', vocab.token2idx.get('<>', 1))
    end_token = vocab.token2idx.get('<PATH_END>', vocab.token2idx.get('</>', 2))

  
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_token_acc': [],
        'val_token_acc': [],
        'test_token_acc': [],
        'train_seq_acc': [],
        'val_seq_acc': [],
        'test_seq_acc': []
    }
    start_epoch = 0
    best_val_loss = float('inf')

    print("="*70)
    print("STARTING NEW TRAINING")
    print("="*70)

    print(f"\nCheckpoints will be saved to: {checkpoint_dir}/")
    print(f"Start token: {start_token} ({vocab.idx2token[start_token]})")
    print(f"End token: {end_token} ({vocab.idx2token[end_token]})")
    print("="*70)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # ====================================================================
        # TRAINING
        # ====================================================================
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, vocab)
        metrics['train_loss'].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # Calculate training accuracies
        train_token_acc, train_seq_acc = calculate_accuracies(
            model, train_loader, vocab, device, start_token, end_token
        )
        metrics['train_token_acc'].append(train_token_acc)
        metrics['train_seq_acc'].append(train_seq_acc)
        print(f"Train Token Acc: {train_token_acc*100:.2f}% | Train Seq Acc: {train_seq_acc*100:.2f}%")

        # ====================================================================
        # VALIDATION
        # ====================================================================
        val_loss = evaluate(model, val_loader, criterion, device, vocab)
        metrics['val_loss'].append(val_loss)
        print(f"Val Loss: {val_loss:.4f}")

        # Calculate validation accuracies
        val_token_acc, val_seq_acc = calculate_accuracies(
            model, val_loader, vocab, device, start_token, end_token
        )
        metrics['val_token_acc'].append(val_token_acc)
        metrics['val_seq_acc'].append(val_seq_acc)
        print(f"Val Token Acc: {val_token_acc*100:.2f}% | Val Seq Acc: {val_seq_acc*100:.2f}%")

        # ====================================================================
        # TEST
        # ====================================================================
        test_loss = evaluate(model, test_loader, criterion, device, vocab)
        metrics['test_loss'].append(test_loss)
        print(f"Test Loss: {test_loss:.4f}")

        # Calculate test accuracies
        test_token_acc, test_seq_acc = calculate_accuracies(
            model, test_loader, vocab, device, start_token, end_token
        )
        metrics['test_token_acc'].append(test_token_acc)
        metrics['test_seq_acc'].append(test_seq_acc)
        print(f"Test Token Acc: {test_token_acc*100:.2f}% | Test Seq Acc: {test_seq_acc*100:.2f}%")

        # ====================================================================
        # SAVE CHECKPOINT
        # ====================================================================
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
            'train_token_acc': train_token_acc,
            'train_seq_acc': train_seq_acc,
            'val_token_acc': val_token_acc,
            'val_seq_acc': val_seq_acc,
            'test_token_acc': test_token_acc,
            'test_seq_acc': test_seq_acc,
        }, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'transformer.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

        print()

    print("="*70)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final epoch: {num_epochs}")
    print("="*70)

    return metrics

def plot_training_curves(metrics, save_prefix='transformer_training'):
    """
    Plot training curves and save as 3 SEPARATE image files:
    1. Loss curves
    2. Token accuracy curves
    3. Sequence accuracy curves
    """
    epochs = range(1, len(metrics['train_loss']) + 1)

    # ========================================================================
    # PLOT 1: LOSS CURVES
    # ========================================================================
    print("\n Creating Loss Curves...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss',
             linewidth=2, markersize=5)
    ax1.plot(epochs, metrics['val_loss'], 'g-s', label='Val Loss',
             linewidth=2, markersize=5)
    ax1.plot(epochs, metrics['test_loss'], 'r-^', label='Test Loss',
             linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    plt.tight_layout()
    loss_path = f'{save_prefix}_loss.png'
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {loss_path}")
    plt.show()
    plt.close()

    # ========================================================================
    # PLOT 2: TOKEN ACCURACY CURVES
    # ========================================================================
    print("\n Creating Token Accuracy Curves...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(epochs, [acc*100 for acc in metrics['train_token_acc']],
             'b-o', label='Train Token Acc', linewidth=2, markersize=5)
    ax2.plot(epochs, [acc*100 for acc in metrics['val_token_acc']],
             'g-s', label='Val Token Acc', linewidth=2, markersize=5)
    ax2.plot(epochs, [acc*100 for acc in metrics['test_token_acc']],
             'r-^', label='Test Token Acc', linewidth=2, markersize=5)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Token Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Token Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)

    plt.tight_layout()
    token_path = f'{save_prefix}_token_accuracy.png'
    plt.savefig(token_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {token_path}")
    plt.show()
    plt.close()

    # ========================================================================
    # PLOT 3: SEQUENCE ACCURACY CURVES
    # ========================================================================
    print("\nCreating Sequence Accuracy Curves...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(epochs, [acc*100 for acc in metrics['train_seq_acc']],
             'b-o', label='Train Seq Acc', linewidth=2, markersize=5)
    ax3.plot(epochs, [acc*100 for acc in metrics['val_seq_acc']],
             'g-s', label='Val Seq Acc', linewidth=2, markersize=5)
    ax3.plot(epochs, [acc*100 for acc in metrics['test_seq_acc']],
             'r-^', label='Test Seq Acc', linewidth=2, markersize=5)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Sequence Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Sequence Accuracy Curves', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs)

    plt.tight_layout()
    seq_path = f'{save_prefix}_sequence_accuracy.png'
    plt.savefig(seq_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {seq_path}")
    plt.show()
    plt.close()

    print(f"\nAll 3 plots saved successfully!")
    print(f"    Files created:")
    print(f"      1. {loss_path}")
    print(f"      2. {token_path}")
    print(f"      3. {seq_path}")



if __name__ == "__main__":
    # Transformer hyperparameters
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 512
    DROPOUT = 0.1

    # Other hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================================
    # Load Data and Create Train/Val Split (80-20)
    # ============================================================================

    train_path = '/content/train_6x6_mazes.csv'
    test_path = '/content/test_6x6_mazes.csv'

    # Load full training dataset
    full_train_dataset = MazeDataset(train_path)
    test_dataset = MazeDataset(test_path, vocab=full_train_dataset.vocab, is_train=False)

    # Split into train (80%) and val (20%)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Create shuffled indices
    indices = list(range(len(full_train_dataset)))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)

    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subset datasets
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Save the pkl file for inference time use
    vocab_size = len(full_train_dataset.vocab)
    with open ("vocab_transformer.pkl", "wb") as f:
        pickle.dump(full_train_dataset.vocab, f)
        
    print("="*60)
    print("DATA LOADING SUMMARY")
    print("="*60)
    print(f"Vocabulary size: {vocab_size}")
    print(f"\nDataset splits:")
    print(f"  Total training data: {len(full_train_dataset)} samples")
    print(f"  Train split: {train_size} samples (80%)")
    print(f"  Val split: {val_size} samples (20%)")
    print(f"  Test set: {len(test_dataset)} samples")
    print(f"\nDataLoader batches (batch_size={BATCH_SIZE}):")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    print("="*60)

    model = TransformerMazeSolver(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        device=DEVICE
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    metrics = train_with_checkpoints(
        model=model, 
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer, 
        vocab=full_train_dataset.vocab,
        device=DEVICE,
        num_epochs=20,
        checkpoint_dir='checkpoints',
    )

    print("\n✓ Training completed!")
    # Plot the curves
    plot_training_curves(metrics, save_prefix='transformer_training')