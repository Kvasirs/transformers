import torch
import numpy as np

def adjust_decoder_attention_2(attention_weights, tweaks):

    np_weights = np.array(attention_weights.cpu())

    # For each beam and and head.
    for b in range(len(np_weights)):
        for h in range(len(np_weights[b])):

            # Get all encdec attention weights.
            weights = np_weights[b][h][0]
            weight_sum = np.sum(weights)

            # For each tweak and index.
            for tweak in tweaks:
                for index in tweak["indexes"]:

                    # Get target weight and potential new weight.
                    target_weight = weights[index]
                    new_weight = tweak["value"] / 10
                        
                    # Select all weights (except for weights at word_pos) and normalize to 1.
                    weights = [weights[i] for i in range(len(weights)) if i != index]
                    weights = np.divide(weights, weight_sum)

                    # Multiply by 1 - new_val, and add new value back in.
                    weights = np.multiply(weights, 1 - new_weight)
                    weights = np.insert(weights, index, new_weight)

            # Set recalculated weights.
            np_weights[b][h][0] = weights

    # Convert back to tensor.
    attention_weights = torch.tensor(np_weights)

    return attention_weights
    

def adjust_decoder_attention(attention_weights, tweaks):

    np_weights = attention_weights.numpy()

    # For each beam and and head.
    for b in range(len(np_weights)):
        for h in range(len(np_weights[b])):

            # Get all encdec attention weights.
            weights = np_weights[b][h][0]

            # For each tweak and its indexes.
            for tweak in tweaks:
                for index in tweak["indexes"]:

                    # Calculate new weight for index.
                    weights[index] = weights[index] * tweak["value"]

            # Normalize to 1.
            weights = np.divide(weights, np.sum(weights))

            # Set recalculated weights.
            np_weights[b][h][0] = weights

    # Convert back to tensor.
    attention_weights = torch.tensor(np_weights)

    return attention_weights

def adjust_att_vecs(q, k, w1_i, w2_i, sim, mag):

    # Convert Q and K tensors to numpy arrays.
    q_np = q.numpy()
    k_np = k.numpy()

    for i in range(len(k_np[0])):

        # Get Q and K vectors we are targetting.
        Q = q_np[0][i][w1_i] 
        K = k_np[0][i][w2_i]

        # Adjust vector using vector interpolate equation.
        K = sim * Q + (1 - sim) * K

        # Change K to its unit vector (magnitude = 1).
        K = K / np.linalg.norm(K)

        # Multiply by specified scalar value to increase magnitude.
        K *= mag

        k_np[0][i][w2_i] = K

        # Print new cosine similarity between Q and K vectors.
        cos_sim = np.dot(Q, K) / (np.linalg.norm(Q) * np.linalg.norm(K))
        #print(cos_sim)

    # Convert K vectors back to tensors.
    k_vecs = torch.tensor(k_np) 

    return k_vecs



