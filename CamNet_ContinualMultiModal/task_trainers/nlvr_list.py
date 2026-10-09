
def extract_nlvr_concept(train_dataset):

    concept_list = []

    for image, pos_sentence, neg_sentence, index in train_dataset:
        pos_words = pos_sentence.split()
        neg_words = neg_sentence.split()
        # print(index, pos_sentence, neg_sentence)
        # Initialize variables to track the differences
        pos_diff = []
        neg_diff = []

        # Find the first different word
        i = 0
        j = 0
        while i < len(pos_words) and j < len(neg_words) and pos_words[i].lower() == neg_words[j].lower():
            i += 1
            j += 1

        # Now i and j point to the first different word
        # Capture the differences from this point onward
        temp_pos_diff = []
        temp_neg_diff = []

        while i < len(pos_words):
            # Check if the remaining part of neg_words contains pos_words[i]
            if pos_words[i].lower() not in [word.lower() for word in neg_words[j:]]:
                temp_pos_diff.append(pos_words[i])
            else:
                # Once we find a match, stop capturing
                break
            i += 1

        while j < len(neg_words):
            # Check if the remaining part of pos_words contains neg_words[j]
            if neg_words[j].lower() not in [word.lower() for word in pos_words[i:]]:
                temp_neg_diff.append(neg_words[j])
            else:
                # Once we find a match, stop capturing
                break
            j += 1

        # If there are multiple elements in temp_pos_diff, join them into a single phrase
        if temp_pos_diff:
            pos_diff.append(" ".join(temp_pos_diff))

        # If there are multiple elements in temp_neg_diff, join them into a single phrase
        if temp_neg_diff:
            neg_diff.append(" ".join(temp_neg_diff))

        concept_list.append({
            'pos_diff': pos_diff,
            'neg_diff': neg_diff
        })
    unique_concept = set()

    for entry in concept_list:
        unique_concept.update(entry['pos_diff'])
        unique_concept.update(entry['neg_diff'])

    # Convert the set to a list
    unique_concept_list = list(unique_concept)

    return unique_concept_list
