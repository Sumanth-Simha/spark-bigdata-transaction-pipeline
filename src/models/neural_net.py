import torch
import torch.nn as nn

class FraudMLP(nn.Module):
    def __init__(self, num_numerical_features, embedding_dims, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        """
        Anomaly detection MLP for tabular data.

        Args:
            num_numerical_features (int): Number of numerical features.
            embedding_dims (list of tuples): List of (num_classes, embedding_dim) for each categorical feature.
            hidden_dims (list of int): Sizes of hidden layers.
            dropout_rate (float): Dropout probability.
        """
        super(FraudMLP, self).__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes, embed_dim in embedding_dims
        ])

        total_embed_dim = sum(embed_dim for _, embed_dim in embedding_dims)
        input_dim = num_numerical_features + total_embed_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer (1 unit for binary classification)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x_num, x_cat):
        # x_num shape: (batch_size, num_numerical_features)
        # x_cat shape: (batch_size, num_categorical_features)

        embeds = []
        for i, emb_layer in enumerate(self.embeddings):
            embeds.append(emb_layer(x_cat[:, i]))

        if embeds:
            x_cat_embed = torch.cat(embeds, dim=1)
            x = torch.cat([x_num, x_cat_embed], dim=1)
        else:
            x = x_num

        x = self.hidden_layers(x)
        out = self.output_layer(x)

        return out
