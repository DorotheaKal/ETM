import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(normd_bow, num_topics, model):
    thetaAvg = torch.zeros(1, num_topics).to(device)
    thetaWeightedAvg = torch.zeros(1, num_topics).to(device)
    sums = normd_bow.sum(1).unsqueeze(1)
    theta, _ = model.get_theta(normd_bow)
    thetaAvg += theta.sum(0).unsqueeze(0) / normd_bow.shape[0]
    weighed_theta = sums * theta
    thetaWeightedAvg += weighed_theta.sum(0).unsqueeze(0)
    thetaWeightedAvg = thetaWeightedAvg.squeeze().cpu().numpy()
    print('\nThe 10 most used topics are {}'.format(thetaWeightedAvg.argsort()[::-1][:10]))
    return thetaWeightedAvg.argsort()[::-1][0]
