from tqdm import tqdm


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()
    losses = []
    for X, y in tqdm(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        try:
            pred = model(X)
        except:
            print(X)
            print(X.shape)
            print(X.min())
            print(X.max())
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Monitoring
        losses.append(loss.item())

    # Returns losses mean
    return sum(losses) / len(losses)
