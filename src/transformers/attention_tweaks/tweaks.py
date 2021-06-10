import torch
import numpy as np

##def adjust_decoder_attention_2(attention_weights, tweaks):
##
##    np_weights = attention_weights.numpy()
##
##    # For each beam and and head.
##    for b in range(len(np_weights)):
##        for h in range(len(np_weights[b])):
##
##            # Get all encdec attention weights.
##            weights = np_weights[b][h][0]
##
##            # Changed sum.
##            new_value_sum = 0
##
##            # Old value sum.
##            old_value_sum = 0
##
##            # Tweaked index list.
##            tweaked_indexes = []
##
##            print("old weights:")
##            print(weights)
##
##            # For each tweak and its indexes.
##            for tweak in tweaks:
##                for index in tweak["indexes"]:
##
##                    old_value_sum += weights[index]
##
##                    # If - value, calculate % decrease (min is 0).
##                    if tweak["value"] <= 0:
##                        weights[index] *= (100 - (-tweak["value"])) / 100
##                    else:
##                        # If + value, increase value by % of diff (1 - new value).
##                        weights[index] += (1 - weights[index]) * tweak["value"] / 100
##
##                    weights[index] /= len(tweak["indexes"])
##
##                    # Add new value to changed sum.
##                    new_value_sum += weights[index]
##
##                    # Add index to index list.
##                    tweaked_indexes.append(index)
##
##            # Once all values have been altered, change
##            # the remaining values to ensure sum = 1.
##            for i in range(len(weights)):
##                if i not in tweaked_indexes:
##                    weights[i] *= ((1 - new_value_sum) / (1 - old_value_sum))
##
##
##            print("new weights:")
##            print(weights)
##            print("\n")
##
##            # Set recalculated weights.
##            np_weights[b][h][0] = weights
##
##    # Convert back to tensor.
##    attention_weights = torch.tensor(np_weights)
##
##    return attention_weights

def adjust_decoder_attention(attention_weights, tweaks):

    print(type(attention_weights)

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



