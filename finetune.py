import transformers
import torch
import datasets
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import torch.optim as optim
import torch.nn.functional as F
import time
import random

# Load cifar10 dataset and extract labels
datasets = load_dataset("cifar10")

labels = datasets["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

#Load model and processor from Hugging Face's transformers library
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# Function to batch large datasets into smaller groups for easier computation
def homemade_batch(num_img, batch_size=10, start_img=0, data_type = "test"):
    # Initialize empty set to store predicted values and their probabilities
    homemade = []
    num_batches = num_img // batch_size
    extra = num_img % batch_size # Not implemented yet

    # Allows computations to be run on GPU instead of CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_0 = time.perf_counter()

    for i in range(num_batches):
        t1 = time.perf_counter()

        # Create a temporary batch of data
        subset = datasets[data_type].select(range((i * batch_size + start_img), (i + 1) * batch_size + start_img))
        input = processor(text=labels, images=subset["img"], return_tensors="pt", padding=False).to(device)
        output = model(**input)

        # Access logits of the input images, apply softmax function
        logits = output.logits_per_image #Finds the similarities
        probs = logits.softmax(dim=1) #Computes the probabilities

        # Find maximum of the probabilities, as well as their corresponding index, append them to list
        max_prob, max_id = probs.max(dim=1)
        homemade.append([max_prob.cpu().detach(), max_id.cpu().detach()])
        torch.cuda.empty_cache()


        t2 = time.perf_counter()
        print(f"Finished batch {i + 1} of {num_batches} in {t2 - t1} seconds")

    t_3 = time.perf_counter()
    print(f"Finished entire dataset in {t_3 - t_0} seconds")

    # Returns list of tensors, structure is [[tensor([first batch maximum probabilities]), tensor([corresponding indices/labels])],
    #                                         [tensor([second batch maximum probabilities]), tensor([corresponding indices/labels])],
    #                                         [tensor([third batch maximum probabilities]), tensor([corresponding indices/labels])]]
    return homemade


# Takes output of homemade_batch as input and returns clean data
def prediction_reformat(subset):
    # Initialize empty list to store new reformatted data
    predicted = []
    count = 0

    # len(subset) = number of batches
    for i in range(len(subset)):
        for k in range(len(subset[0][0])):
            prob = subset[i][0][k].item()
            id = subset[i][1][k].item()

            label = id2label[id]
            predicted.append([count, label, prob, id])

            count += 1

    # Returns nested list with form [[index, "label", probability, id],
    #                                [index, "label", probability, id]]
    return predicted


# Computes how accurate the model is
def accuracy(result):
    correct = 0
    total = 0

    # Create dictionary to count how many of each label occurs in the subset, all labels initialized to 0
    all_labels = {}
    for label in datasets["train"].features["label"].names:
        all_labels[label] = 0

    # Dictionary to keep track of which classes were incorrectly predicted
    incorrect = {}
    for label in datasets["train"].features["label"].names:
        incorrect[label] = 0

    # Iterate through the results for each image in the subset
    for i in range(len(result)):
        # Automatically increase count of label in dictionary for appearing
        all_labels[result[i][1]] += 1

        # If the actual id/label aligns with the predicted one, add to correct count
        if datasets["train"][i]["label"] == result[i][3]:
            correct += 1
            total += 1
            if total % 50 == 0:
              print(f"Model accurately predicted {result[i][1]} with {result[i][2] * 100}% confidence.")
        else:
            # If they do not align, increase count of predicted id/label in incorrect dictionary
            total += 1
            print(f"Model inaccurately predicted {result[i][1]} with {result[i][2] * 100}% confidence.")
            incorrect[result[i][1]] += 1

    print(f"Accuracy: {(correct/total) * 100}%")

    worst_accuracy = []
    # For every label, calculate percentage predicted correctly by subtracting total by incorrect
    for label in all_labels:
        correct =  all_labels[label] - incorrect[label]
        total = all_labels[label]
        print(f"For {label}: Predicted {correct} out of {total} correct. {(correct) / total * 100}% Accuracy")
        worst_accuracy.append([label, correct/total])

    worst_group = min(worst_accuracy, key=lambda x: x[1])
    print(f"The worst performing group is '{worst_group[0]}' with an accuracy of {worst_group[1] * 100}%")


#Combines the previous 2 functions together 
def data_analysis(predictions, data_type = "test"):
    cleaned = prediction_reformat(predictions)
    final_results = accuracy(cleaned, data_type)

    return final_results



# Train model on training dataset
def train(num_img, batch_size=10, num_epoch=2):
    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    total_loss = 0
    model.to(device)
    model.train()


    for epoch in range(num_epoch):
        t0 = time.perf_counter()
        epoch_loss = 0

        # Separate training data into smaller batches
        for i in range(num_img // batch_size):
            train_set = datasets["train"].select(range(i * batch_size, (i + 1) * batch_size))
            t1 = time.perf_counter()

            # Process the data, feed it into the model
            input = processor(text=labels, images=train_set["img"], return_tensors="pt", padding=False).to(device)
            output = model(**input)

            # Get the logit for the predictions on the image and text
            logits_per_image = output.logits_per_image
            logits_per_text = output.logits_per_text
            # Turn this tensor from batch_size x 1 matrix to 1 x batch_size (doesn't work otherwise)
            logits_per_text = logits_per_text.squeeze()

            # Accesses the ground truth
            targets = torch.tensor(train_set["label"]).to(device)

            # Uses the cross-loss entropy function to calculate the loss of the images and text, utilizes softmax activation
            loss_img = F.cross_entropy(logits_per_image, targets)
            loss_text = F.cross_entropy(logits_per_text, targets)

            # Calculate the total loss 
            loss = (loss_img + loss_text) / 2
            t2 = time.perf_counter()
            print(f"Finished batch {i + 1}/{num_img // batch_size} in {t2 - t1} seconds")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        t3 = time.perf_counter()
        total_loss += epoch_loss
        avg_loss = epoch_loss / (num_img // batch_size)
        print(f"Epoch {epoch+1}/{num_epoch} completed in {t3 - t1} seconds, Loss: {avg_loss:.4f}")


# Shuffle the dataset to prevent overfitting
def train_shuffled(num_img, batch_size=10, num_epoch=2):
    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    total_loss = 0
    model.to(device)
    model.train()

    for epoch in range(num_epoch):
        t0 = time.perf_counter()
        epoch_loss = 0

        # Shuffle the entire training dataset once per epoch
        shuffled_dataset = datasets["train"].shuffle(seed=random.randint(0, 1000))

        # Separate training data into smaller batches
        for i in range(num_img // batch_size):
            # Select the current batch from the shuffled dataset
            train_set = shuffled_dataset.select(range(i * batch_size, (i + 1) * batch_size))
            t1 = time.perf_counter()

            # Process the data, feed it into the model
            input = processor(text=labels, images=train_set["img"], return_tensors="pt", padding=False).to(device)
            output = model(**input)

            # Get the logits for predictions on the image and text
            logits_per_image = output.logits_per_image
            logits_per_text = output.logits_per_text.squeeze()  # Make sure it's the correct shape

            # Access ground truth
            targets = torch.tensor(train_set["label"]).to(device)

            # Calculate loss
            loss_img = F.cross_entropy(logits_per_image, targets)
            loss_text = F.cross_entropy(logits_per_text, targets)
            loss = (loss_img + loss_text) / 2
            t2 = time.perf_counter()
            print(f"Finished batch {i + 1}/{num_img // batch_size} in {t2 - t1} seconds")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        total_loss += epoch_loss
        avg_loss = epoch_loss / (num_img // batch_size)
        t3 = time.perf_counter()
        print(f"Epoch {epoch+1}/{num_epoch} completed in {t3 - t0} seconds, Loss: {avg_loss:.4f}")



