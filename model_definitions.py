import torch
import torch.nn as nn

class ParticleDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, weights=None):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.x_data[idx], self.y_data[idx], self.weights[idx]
        return self.x_data[idx], self.y_data[idx]

class ParticleClassifier(nn.Module):
    def __init__(self, input_shape=(5, 30)):
        super(ParticleClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # Shape: [batch, 32, n_particles]
        x = x.transpose(1, 2)    # Shape: [batch, n_particles, 32]
        
        # Attention mechanism
        attention_weights = self.attention(x)  # Shape: [batch, n_particles, 1]
        x = attention_weights * x             # Apply attention weights
        x = torch.sum(x, dim=1)               # Sum over particles
        
        # Classification
        x = self.classifier(x)
        return x

class ParticleTransformer(nn.Module):
    def __init__(self, input_dim=5, max_particles=30, embed_dim=64, num_heads=4, 
                 ff_dim=128, num_layers=3, dropout=0.2, particle_embedding_dim=16):
        super(ParticleTransformer, self).__init__()
        
        self.input_dim = input_dim - 1  # Momentum features (excluding particle type)
        self.max_particles = max_particles
        self.embed_dim = embed_dim
        
        # Particle type embedding
        self.particle_embedding = nn.Embedding(
            num_embeddings=7,  # 7 values: 0,1,2,3,4,5,6 (shifted from -1,0,1,2,3,4,5)
            embedding_dim=particle_embedding_dim
        )        

        # Momentum feature embedding
        self.momentum_embedding = nn.Linear(self.input_dim, embed_dim - particle_embedding_dim)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input shape: [batch_size, num_features, max_particles]
        # Transpose to [batch_size, max_particles, num_features]
        x = x.transpose(1, 2)
        
        # Split momentum features and particle type
        momentum_features = x[:, :, :4]  # First 4 features (px, py, pz, e)
        particle_types = x[:, :, 4].long()  # 5th feature is particle type
        
        # Shift particle types from (-1 to 5) to (0 to 6)
        particle_types = particle_types + 1
        
        # Safeguard against out of bounds values
        particle_types = torch.clamp(particle_types, min=0, max=6)
        
        # Get embeddings
        type_embeddings = self.particle_embedding(particle_types)
        momentum_embeddings = self.momentum_embedding(momentum_features)
        
        # Concatenate embeddings
        x = torch.cat([momentum_embeddings, type_embeddings], dim=2)
        
        # Create mask for padding (zero values)
        padding_mask = (torch.sum(momentum_features.abs(), dim=2)) < 1e-6
        
        # Apply transformer encoder with padding mask
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global pooling of non-padding tokens
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        x_sum = (transformer_output * mask_expanded).sum(dim=1)
        token_count = mask_expanded.sum(dim=1) + 1e-10  # Avoid division by zero
        x_mean = x_sum / token_count
        
        # Apply classification head
        output = self.classifier(x_mean)

        return output