from metrics import *
def train_a_model(model, loss, optimizer, training_data, train_for_evaluation, validation, params_path, epochs=10):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  metric_train_history = []
  metric_test_history = []
  loss_history = []
  criterion = loss.to(device)
  scaler = torch.amp.GradScaler('cuda')

  #loss = torch.tensor(100.0) . Current loss: {loss.item()}
  for ep in range(epochs):
    model.train()
    for i, (X, y) in enumerate(tqdm(training_data, desc=f'Epoch {ep+1}')):
      X = X.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)
      X = X.unsqueeze(1)

      with torch.amp.autocast(device_type='cuda'):
        pred = model(X).squeeze()
        loss = criterion(pred, y)
        loss_history.append(loss.item())

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()

      #if i % 5 == 4:
      #  gc.collect()

    gc.collect()
    torch.cuda.empty_cache()

    metric_train_history = evaluate(model, train_for_evaluation, device)
    metric_test_history = evaluate(model, validation,  device)

    gc.collect()
    torch.cuda.empty_cache()

    param_dist = params_path + f'/ep{ep+1}_params.pt'
    torch.save(model.state_dict(), param_dist)

  return model, loss_history, metric_train_history, metric_test_history