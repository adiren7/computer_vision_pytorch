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

            y_pred = model(X).squeeze()

            y_proba = torch.softmax(y_pred,dim=0)

            ç, y_label = torch.max(y_proba,1)

            ground_truth.extend(y.cpu().numpy())
            predictions.extend(y_label.cpu().numpy())



    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    cm = confusion_matrix(ground_truth, predictions)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return precision , recall , f1