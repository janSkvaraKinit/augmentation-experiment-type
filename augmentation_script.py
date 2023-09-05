from datasets import load_dataset
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import random
import numpy
from deep_translator import GoogleTranslator
import pandas as pd

def synonym_augmentation(text, synonym_aug_p):
  aug = naw.SynonymAug(aug_src='wordnet', aug_p=synonym_aug_p , aug_min=1, aug_max=None)
  return aug.augment(text)

def swap_word_augmentation(text, swap_word_aug_p):
  aug = naw.RandomWordAug(action="swap", aug_p=swap_word_aug_p, aug_min=1, aug_max=None)
  return aug.augment(text)

def delete_word_augmentation(text, delete_word_aug_p):
  aug = naw.RandomWordAug(action="delete", aug_p=delete_word_aug_p, aug_min=1, aug_max=None)
  return aug.augment(text)

def back_translation_augmentation(text, original_language, target_language ):
  translator = GoogleTranslator(source=original_language, target=target_language)
  back_translator = GoogleTranslator(source=target_language, target=original_language)

  translation = translator.translate(text)
  return back_translator.translate(translation)

def swap_neighbor_sentences_augmentation(text, swap_neighbor_sentences_aug_p):
  aug = nas.RandomSentAug(mode='neighbor', action='swap', aug_p=swap_neighbor_sentences_aug_p, aug_min=1, aug_max=None)
  return aug.augment(text)

def swap_random_sentences_augmentation(text, swap_random_sentences_aug_p):
  aug = nas.RandomSentAug(mode='random', action='swap', aug_p=swap_random_sentences_aug_p, aug_min=1, aug_max=None)
  return aug.augment(text)

def main():
  name_of_dataset = "imdb" #Name of dataset from Hagging Face
  seed = 10 #Random seed, by default the random number generator uses the current system time
  proportio_of_dataset = 0.01 #Proportion of dataset, which will be augmentated
  split_name = "train" #Split name, which indicate what loaded dataset we should augmentate

  synonym_aug_probability = 1 #Probability distribution for synonym augmentation
  synonym_aug_p = 0.3 #Percentage of word will be synonym word augmented 

  swap_word_aug_probability = 0 #Probability distribution for swap word augmentation
  swap_word_aug_p = 0.3 #Percentage of word will be randomly swapped

  delete_word_aug_probability = 0 #Probability distribution for delete word augmentation
  delete_word_aug_p = 0.3 #Percentage of word will be randomly deleted
  
  swap_neighbor_sentences_aug_probability = 0 #Probability distribution for delete word augmentation
  swap_neighbor_sentences_aug_p = 0.3 #Percentage of sentence will be augmented
  
  swap_random_sentences_aug_probability = 0 #Probability distribution for delete word augmentation
  swap_random_sentences_aug_p = 0.3 #Percentage of sentences will be deleted

  back_translation_aug_probability = 0 #Probability distribution for back translation augmentation
  original_lan_back_translation_aug = 'en' #Original language of text
  target_lan_back_translation_aug = 'sk' #Target language, which is used during back transaltion augmentation

  path_and_name_of_generated_csv_file = 'filename.csv'

  dataset = load_dataset(name_of_dataset)

  if 1 != sum([synonym_aug_probability, swap_word_aug_probability, delete_word_aug_probability, swap_neighbor_sentences_aug_probability, swap_random_sentences_aug_probability, back_translation_aug_probability]):
    print("Summ of probabilities MUST be 1!")
    return

  if(seed == 0):
      random.seed()
  else:
      random.seed(seed)

  size_of_augmented_dataset = round(len(dataset[split_name]) * proportio_of_dataset)
  print(f"Size of augmented dataset: {size_of_augmented_dataset}")
  dataset_indices = random.sample(range(len(dataset[split_name])), size_of_augmented_dataset)
  augmentation_samples = dataset[split_name][dataset_indices]

  csv_list = []
  for index in range(len(augmentation_samples['text'])):
    print(index)
    augmented_text = str()
    text = augmentation_samples['text'][index]
    augmetation_method = numpy.random.choice(6, 1, p=[synonym_aug_probability, swap_word_aug_probability, delete_word_aug_probability, 
    swap_neighbor_sentences_aug_probability, swap_random_sentences_aug_probability, back_translation_aug_probability])

    #Word augmentation methods
    if augmetation_method == 0:
      augmented_text = synonym_augmentation(text, synonym_aug_p)
    elif augmetation_method == 1:
      augmented_text = swap_word_augmentation(text, swap_word_aug_p)
    elif augmetation_method == 2:
      augmented_text = delete_word_augmentation(text, delete_word_aug_p)
    
    #Sentence augmentation
    elif augmetation_method == 3:
      augmented_text = swap_neighbor_sentences_augmentation(text, swap_neighbor_sentences_aug_p)
    elif augmetation_method == 4:
      augmented_text =swap_random_sentences_augmentation(text, swap_random_sentences_aug_p)

    #Others
    elif augmetation_method == 5:
      augmented_text = back_translation_augmentation(text, original_lan_back_translation_aug, target_lan_back_translation_aug)

    new_row = {'text': augmentation_samples['text'][index], 'label': augmentation_samples['label'][index], 'augmented_text': augmented_text[0]}
    csv_list.append(new_row)

  df = pd.DataFrame.from_records(csv_list)
  df.to_csv(path_and_name_of_generated_csv_file, index=False, sep=";")

if __name__ == "__main__":
  main()