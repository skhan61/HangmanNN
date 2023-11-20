import random

def group_words_by_length(word_list):
    word_groups = {}
    for word in word_list:
        word_length = len(word)
        if word_length not in word_groups:
            word_groups[word_length] = []
        word_groups[word_length].append(word)
    return word_groups

def sample_words_stratified(word_list, word_frequencies, num_samples):
    word_groups = group_words_by_length(word_list)
    total_words = sum(len(group) for group in word_groups.values())
    sampled_words = []

    print(f"Total words: {total_words}")
    print(f"Number of groups: {len(word_groups)}")

    for word_length, words in word_groups.items():
        words_sorted_by_frequency = sorted(words, key=lambda w: word_frequencies.get(w, 0))
        high_freq_words = words_sorted_by_frequency[:len(words)//2]
        low_freq_words = words_sorted_by_frequency[len(words)//2:]

        num_samples_group = max(1, int(num_samples * len(words) / total_words))

        # print(f"Word Length: {word_length}, Group Size: {len(words)}, Samples for Group: {num_samples_group}")

        if len(words) > 0:
            high_freq_samples = min(len(high_freq_words), num_samples_group // 2)
            low_freq_samples = min(len(low_freq_words), num_samples_group // 2)
            sampled_high_freq_words = random.sample(high_freq_words, high_freq_samples)
            sampled_low_freq_words = random.sample(low_freq_words, low_freq_samples)

            # print(f"    High Freq Samples: {high_freq_samples}, Low Freq Samples: {low_freq_samples}")
            # print(f"    Sampled High Freq Words: {sampled_high_freq_words}")
            # print(f"    Sampled Low Freq Words: {sampled_low_freq_words}")

            sampled_words.extend(sampled_high_freq_words)
            sampled_words.extend(sampled_low_freq_words)

    return sampled_words

# Rest of your code for testing

num_samples = 200000

sampled_words = sample_words_stratified(word_list, word_frequencies, num_samples)
print(len(sampled_words))

def generate_masked_word_variants(word, max_variants):
    word_length = len(word)
    indices = list(range(word_length))
    masked_versions = set()

    # Define a reasonable cap for the number of masks based on word length
    max_masks = min(word_length // 2, 10)  # Adjust this as needed

    # Limit the number of variants directly based on max_variants
    while len(masked_versions) < max_variants:
        num_masks = random.randint(1, max_masks)
        mask_indices = set(random.sample(indices, num_masks))
        masked_word = ''.join(c if i not in mask_indices else '_' for i, c in enumerate(word))
        masked_versions.add(masked_word)

    return list(masked_versions)

def process_word(word):
    word_length = len(word)
    # Directly relate the max number of variants to the word length
    max_variants = min(word_length // 2, 10)  # Adjust this formula as needed
    return generate_masked_word_variants(word, max_variants)


from multiprocessing import Pool

if __name__ == "__main__":
    pool = Pool(processes=4)  # Adjust the number of processes based on your system
    all_masked_versions = pool.map(process_word, sampled_words)
    pool.close()
    pool.join()


# max_example_from_word = 5 
# for word in sampled_words:
#     masked_versions = generate_masked_word_variants(word, max_example_from_word)
#     # print(masked_versions)
#     # break

# def process_word(word):
#     max_examples_from_word = 5
#     return generate_masked_word_variants(word, max_examples_from_word)

# if __name__ == "__main__":
#     pool = Pool(processes=4)  # Adjust the number of processes based on your system
#     all_masked_versions = pool.map(process_word, sampled_words)
#     pool.close()
#     pool.join()
