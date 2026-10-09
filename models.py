import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyg_compat import GCNConv, GINConv, GATv2Conv, GraphNorm, JumpingKnowledge
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SimpleGCN(nn.Module):
    # def __init__(self, input_dim, hidden_dim, output_dim):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = x[:, :-1]
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    
class StackedGATv2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers, num_heads=4, jk_mode='last', dropout=0.2):
        super(StackedGATv2, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            num_heads = 1 if i == num_layers - 1 else num_heads

            self.convs.append(
                GATv2Conv(in_channels=in_dim,
                          out_channels=out_dim // num_heads,
                          heads=num_heads,
                          dropout=dropout,
                          residual=True,
                          edge_dim=None)  # set to your edge_attr dim if available
            )
            self.norms.append(GraphNorm(out_dim))

        if jk_mode == 'last':
            self.jk = None
        elif jk_mode == 'lstm':
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=num_layers)
        else:
            self.jk = JumpingKnowledge(mode=jk_mode)

    def forward(self, x, edge_index, batch, edge_attr=None):
        out_list = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x, batch)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                out_list.append(x)

        if self.jk:
            return self.jk(out_list)
        else:
            return x
        
        


class TrialGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, jk_mode='last', dropout=0.2):
        super(TrialGIN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(GraphNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(GraphNorm(hidden_dim))

        # Output layer
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(output_dim, output_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(GraphNorm(output_dim))

        if jk_mode == 'last':
            self.jk = None
        elif jk_mode == 'lstm':
            self.jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=num_layers)
        else:
            self.jk = JumpingKnowledge(mode=jk_mode)

    def forward(self, x, edge_index, batch):
        out_list = []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                out_list.append(x)

        if self.jk:
            return self.jk(out_list)
        else:
            return x
        


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, feedforward_dim, num_heads, num_layers, num_classes, pos_enc_dim, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Mapping node features to hidden_dim
        self.mask_embedding = nn.Parameter(torch.randn(hidden_dim))  # Learnable mask embedding
        # Layer normalization for input and positional encodings
        self.ln_input = nn.LayerNorm(hidden_dim)
        self.ln_pos = nn.LayerNorm(hidden_dim)
        # Transformer encoder layers
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=feedforward_dim, dropout=dropout, batch_first=True) # batch_first=True: input shape: [batch_size, seq_len, d_model]
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.pos_enc_project = nn.Linear(pos_enc_dim, hidden_dim) # Project the positional encoding to the same dimension as the node features
        # Prediction heads
        self.fc_type = nn.Linear(hidden_dim, num_classes)  # Node type classification

    @staticmethod
    def create_attention_mask(batch: torch.Tensor) -> torch.Tensor:
        """
        Create a boolean attention mask for disjoint graphs in a batch.
        batch: [num_nodes] indicating the graph index for each node
        Returns: [num_nodes, num_nodes] where True blocks attention (different graphs)
        """
        return batch.unsqueeze(1) != batch.unsqueeze(0) # True for different graphs
    

    def forward(self, x, batch, pos_enc=None, mask=None):
        """
        x: Input node features of shape [num_nodes, input_dim]
        batch: A tensor that stores the batch indices for each node (shape: [num_nodes])
        pos_enc: Laplacian eigenvector positional encoding of shape [num_nodes, pos_enc_dim]
        mask: A binary mask of shape [num_nodes], where 1 means the node is masked and 0 means unmasked
        Returns:
            - embeddings: [num_nodes, hidden_dim] - node embeddings after transformer
        """
        # Step 1: Apply the initial embedding to the input node features
        x = self.embedding(x)  # Convert input node features to hidden_dim

        # Step 2: Apply the mask embedding where the mask is True (i.e., for masked nodes)
        if mask is not None:
            mask_embedding = self.mask_embedding.expand(x.shape[0], -1)  # Expand to match the batch size
            x[mask == 1] = mask_embedding[mask == 1]  # Replace masked nodes' embeddings with mask embedding

        # Normalize both x and positional encoding
        x = self.ln_input(x)
        if pos_enc is not None:
            pos_enc_proj = self.ln_pos(self.pos_enc_project(pos_enc))
            x = x + pos_enc_proj
        else:
            print("No positional encoding provided")

        # Step 3: Create the custom attention mask
        bool_mask = self.create_attention_mask(batch)  # [num_nodes, num_nodes], dtype=bool
        attention_mask = torch.zeros_like(bool_mask, dtype=torch.float32)
        attention_mask[bool_mask] = float('-inf')  # fill only disallowed positions with -inf
        attention_mask = attention_mask.to(x.device)

        # Step 4: Apply the transformer encoder
        x = self.transformer_encoder(x.unsqueeze(0), mask=attention_mask)

        # Step 5: Return node embeddings
        x = x.squeeze(0) # Remove the batch dimension

        return x

    def get_logits(self, embeddings):
        """
        Convert embeddings to logits for node type prediction.
        embeddings: [num_nodes, hidden_dim]
        Returns:
            - type_logits: [num_nodes, num_classes]
        """
        type_logits = self.fc_type(embeddings)     # [num_nodes, num_classes]
        return type_logits


class StaticWidthBitEmbedding(nn.Module):
    def __init__(self, num_bits=32, emb_dim=32):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(num_bits, emb_dim)) # Learnable embedding vectors
        self.max_width = num_bits
        self.emb_dim = emb_dim

    def forward(self, widths):
        """
        Convert a batch of node widths (scalars) to corresponding embeddings.
        widths: tensor of shape [batch_size], each element is a node width (e.g., [13, 8, 16, ...])
        return: tensor of shape [batch_size, emb_dim], corresponding sum of embeddings for each width
        """
        # Ensure widths are integers
        widths = widths.int()

        # Prepare an output tensor to hold the embeddings for the batch
        batch_size = widths.size(0)
        batch_embeddings = torch.zeros(batch_size, self.emb_dim, device=widths.device)

        for i in range(batch_size):
            width = widths[i]
            if width > 0:
                if width > self.max_width:
                    width = self.max_width  # Cap the width to avoid exceeding the number of embeddings
                batch_embeddings[i] = self.embeddings[:width].sum(dim=0)

        return batch_embeddings


class WidthEmbedding(nn.Module):
    """Encode log-scaled positive widths without capping buses at 32 bits."""
    def __init__(self, emb_dim=32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, widths):
        return self.network(torch.log2(1 + widths.float()).unsqueeze(-1))
    

class CdfgNN(nn.Module):
    def __init__(self, 
                 gnn_type, gnn_input_dim, gnn_hidden_dim, gnn_output_dim, num_gnn_layers, gnn_num_heads, jk_mode, gnn_dropout,
                 transformer_hidden_dim, transformer_feedforward_dim, transformer_num_heads, num_transformer_layers, transformer_dropout,
                 num_classes, pos_enc_dim, delete_node_width_embedding, edge_predictor_input_dim,
                 num_bits=32, emb_dim=32):
        super().__init__()

        self.delete_node_width_embedding = delete_node_width_embedding

        if self.delete_node_width_embedding:
            gnn_input_dim = gnn_input_dim - emb_dim

        # Embed node width
        self.width_embedder = WidthEmbedding(emb_dim=emb_dim)

        # Stage 1: GNN
        if gnn_type == "gin":
            self.gnn = TrialGIN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim, num_gnn_layers, jk_mode, gnn_dropout)
        elif gnn_type == "gatv2":
            self.gnn = StackedGATv2(gnn_input_dim, gnn_hidden_dim, gnn_output_dim, num_gnn_layers, num_heads=gnn_num_heads, jk_mode=jk_mode, dropout=gnn_dropout)
        else:
            raise ValueError(f"Invalid GNN type: {gnn_type}")

        # Stage 2: Transformer
        self.transformer = TransformerModel(
            input_dim=gnn_output_dim,       # GNN output goes into transformer
            hidden_dim=transformer_hidden_dim,
            feedforward_dim=transformer_feedforward_dim,
            num_heads=transformer_num_heads,
            num_layers=num_transformer_layers,
            num_classes=num_classes,
            pos_enc_dim=pos_enc_dim,
            dropout=transformer_dropout
        )

        # Stage 3: Edge predictor - single layer for edge prediction task
        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_predictor_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # Binary classification for edge existence
        )

    def encode_nodes(self, x, edge_index, batch, pos_enc, mask=None):
        """
        x: [num_nodes, input_dim] - original node features
        edge_index: [2, num_edges] - COO (coordinate format) format
        batch: [num_nodes] - batch indices for each node
        pos_enc: [num_nodes, k] - Laplacian eigenvector positional encoding
        mask: [num_nodes] binary mask (1 = masked node, 0 = unmasked node)
        """
        
        if self.delete_node_width_embedding:
            x = x[:, :-1]
        else:
            # Extract the node width (last column of the feature vector)
            node_width = x[:, -1]
            
            # Pass the batch of node widths through StaticWidthBitEmbedding
            width_embedding = self.width_embedder(node_width)

            # Remove the width column and concatenate the width embedding to node features
            x = x[:, :-1]  # Remove the last column (node width)
            x = torch.cat([x, width_embedding], dim=-1)  # Concatenate width embedding to the original features

        # Step 1: GNN embedding
        gnn_out = self.gnn(x, edge_index, batch)  # [num_nodes, gnn_output_dim]

        # Step 2: Transformer to get node embeddings
        embeddings = self.transformer(gnn_out, batch, pos_enc, mask)

        return embeddings

    def forward(self, x, edge_index, batch, pos_enc, mask=None):
        embeddings = self.encode_nodes(x, edge_index, batch, pos_enc, mask)
        return self.transformer.get_logits(embeddings)
