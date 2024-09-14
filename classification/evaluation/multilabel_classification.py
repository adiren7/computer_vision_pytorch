from sklearn.metrics import confusion_matrix , precision_score, recall_score, f1_score
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, valid_dataloader, device):
    model.eval()

    ground_truth = []
    predictions = []

    with torch.inference_mode():
        for X,y in valid_dataloader:
            X , y = X.to(device) , y.float().to(device)

            y_pred = model(X)

            y_proba = torch.sigmoid(y_pred)

            y_label = (y_proba > 0.5).float()

            ground_truth.extend(y.cpu().numpy())
            predictions.extend(y_label.cpu().numpy())



    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)


    return precision , recall , f1