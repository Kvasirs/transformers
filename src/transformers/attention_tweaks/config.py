# Tweaked positions - contains pos and new value.
tweaked_positions = []

# Selected word pairs for attention adjustment.
first_word_index = 0
second_word_index = 0

# Should be true if adjusting scores not vectors.
tweaking_scores = False
# Updated attention weight value.
updated_value = 0.5
# Attention scores (softmax values).
current_attention_scores = None

# Should be true if adjusting weights not scores.
tweaking_vectors = False
# Updated angle and magnitude.
new_angle = 1
new_magnitude = 7
# Attention query and key vectors.
current_attention_qk_vecs = None


