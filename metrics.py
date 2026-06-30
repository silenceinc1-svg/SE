from pesq import pesq
from pystoi import stoi
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np

sampling_rate = 16_000

def pesq_batch(sr, Y, X):
    X = X.detach().cpu().numpy() if X.is_cuda else X.detach().numpy()
    Y = Y.detach().cpu().numpy() if Y.is_cuda else Y.detach().numpy()

    pesq_score = Parallel(n_jobs=-1)(delayed(pesq)(sr, y, x, on_error=1)
                  for x, y in zip(Y, X))
    pesq_score = np.array(pesq_score)
    return pesq_score

def stoi_batch(Y, X, sr):
  X = X.detach().cpu().numpy() if X.is_cuda else X.detach().numpy()
  Y = Y.detach().cpu().numpy() if Y.is_cuda else Y.detach().numpy()
  stoi_score = Parallel(n_jobs=-1)(delayed(stoi)(y, x, sr)
                  for x, y in zip(Y, X))
  stoi_score = np.array(stoi_score)
  return stoi_score

def inspect_data_and_metrics(data, verbose = True):
  Pesq = [0, 0]
  Stoi = [0, 0]
  l = len(data)
  for X, Y in tqdm(data):
    p = pesq_batch(sampling_rate, Y, X)
    Pesq[0] += p.mean()/l

    s = stoi_batch(Y, X, sampling_rate)
    Stoi[0] += s.mean()/l

    p = pesq_batch(sampling_rate, Y, Y.flip(0))
    Pesq[1] += p.mean()/l

    s = stoi_batch(Y, Y.flip(0), sampling_rate)
    Stoi[1] += s.mean()/l

  if verbose:
    print(f'Noisy  PESQ: {Pesq[0]:.4f}, STOI: {Stoi[0]:.4f}')
    print(f'Random PESQ: {Pesq[1]:.4f}, STOI: {Stoi[1]:.4f}')
    print()

  return Pesq, Stoi

def evaluate(model, evaluating_data, device = torch.device('cpu'), verbose = True):
  model.to(device)
  model.eval()
  #if  device.type == 'cuda':
  #  attenuate_pretrained = torch.compile(attenuate_pretrained, dynamic=True)

  pesq_sum = 0.0
  stoi_sum = 0.0
  for X, y in tqdm(evaluating_data, desc="Evaluating"):
      X = X.to(device, non_blocking=True)

      if X.dim() == 2:
          X = X.unsqueeze(1)

      with torch.no_grad():
          with torch.amp.autocast(device_type='cuda'):
              pred = model(X).squeeze()

          pred_cpu = pred.cpu()

      p_after = pesq_batch(sampling_rate, y, pred_cpu)
      pesq_sum += p_after.mean().item()

      s_after = stoi_batch(y, pred_cpu, sampling_rate)
      stoi_sum += s_after.mean().item()

  i = len(evaluating_data)
  if verbose:
    print(f'    PESQ: {pesq_sum/i:.4f}')
    print(f'    STOI: {stoi_sum/i:.4f}')
  return pesq_sum/i, stoi_sum/i


if __name__ == "__main__":
    from data import *
    inspect_data_and_metrics(shortened_train_data)
    inspect_data_and_metrics(evaluation_data)
    inspect_data_and_metrics(test_data)

    X, Y = test_data.__iter__().__next__()
    p = pesq_batch(sampling_rate, Y, Y)
    print(p.mean(), p.std())
    s = stoi_batch(Y, Y, sampling_rate)
    print(s.mean(), s.std())
