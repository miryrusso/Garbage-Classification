from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import timm, torch.nn as nn
from torch.optim import SGD
import os
import torch, time
from torch.utils.tensorboard import SummaryWriter



def train_validate(garbage_train_loader, garbage_valid_loader, log_dir,
                   early_stop=0, dropout = False,
                   epochs=5, lr=1e-3, momentum=0.9,
                   device=None, log_every=50,
                   resume_from=None):
    writer = SummaryWriter(log_dir=log_dir)   # dashboard live
    global_step = 0                           # contatore campioni visti
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    num_classes = len(class_dict)

    if dropout:
      model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(
        in_features = model.classifier[1].in_features,
        out_features = num_classes))
    else:
      model.classifier[1] = nn.Linear(
        in_features = model.classifier[1].in_features,
        out_features = num_classes)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    best_val_acc = 0
    best_val_loss = 1

    start_epoch = 1

    patience = early_stop
    patience_counter = 0

    # Riprendi da checkpoint se fornito
    if resume_from:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"Ripreso da: {resume_from} | epoca {start_epoch} | best_acc: {best_val_acc:.3f}")

    num_model_classes = model.classifier[-1].out_features
    save_dir = '/content/drive/MyDrive/modelli_garbage'
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(start_epoch, epochs + 1):
        model.train()
        running_loss, running_acc, seen = 0, 0, 0
        t0 = time.time()

        for step, (x, y) in enumerate(garbage_train_loader, 1):
            x = x.to(device)
            y = y.to(device)

            if step == 1 and ep == 1:
                print(f"Label originali: min={y.min()}, max={y.max()}, unique={torch.unique(y)}")
                print(f"Numero classi modello: {num_model_classes}")

            assert y.min() >= 0, f"Label negativa: {y.min()}"
            assert y.max() < num_model_classes, f"Label {y.max()} >= num_classes {num_model_classes}"

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            n = x.size(0)
            global_step += n # [NUOVO]
            writer.add_scalar('loss/train_iter', loss.item(), global_step) # [NUOVO]
            running_loss += loss.item() * n
            running_acc  += (out.argmax(1) == y).sum().item()
            seen += n

            if step % log_every == 0:
                print(f'Epoch {ep} | step {step}/{len(garbage_train_loader)} '
                      f'loss {running_loss/seen:.4f} acc {running_acc/seen:.3f}')
                writer.add_scalar('loss/train_epoch', loss.item(), global_step) # [NUOVO]

        #[NUOVO]
        train_epoch_loss = running_loss / seen
        train_epoch_acc  = running_acc  / seen
        writer.add_scalar('loss/train_epoch', train_epoch_loss, ep)
        writer.add_scalar('acc/train_epoch',  train_epoch_acc,  ep)
        loss_history_train.append(train_epoch_loss)
        acc_history_train.append(train_epoch_acc)
        # VALIDAZIONE
        model.eval()
        val_loss, val_acc, seen = 0, 0, 0
        with torch.no_grad():
            for x, y in garbage_valid_loader:
                x, y = x.to(device), y.to(device)
                if y.min() > 0:
                    y = y - 1

                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                val_acc  += (out.argmax(1) == y).sum().item()
                seen += x.size(0)

        val_loss /= seen
        val_acc  /= seen
        dt = time.time() - t0
        writer.add_scalar('loss/val_epoch', val_loss, ep) # [NUOVO]
        writer.add_scalar('acc/val_epoch',  val_acc,  ep) # [Nuovo]
        loss_history_val.append(val_loss) # [Nuovo]
        acc_history_val.append(val_acc) # [Nuovo]
        print(f'- Epoch {ep} done in {dt:.1f}s | val_loss {val_loss:.4f} val_acc {val_acc:.3f}')

        #SALVA MIGLIOR MODELLO
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'loss': val_loss
            }, f'{save_dir}/best_model.pt')
            print('  ✓ checkpoint salvato (miglior modello)')
        elif (patience > 0):
            patience_counter += 1
            print(f"- Early stopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping attivato.")
                break

        #SALVA OGNI 2 EPOCHE PERCHE VOLEVO PROVARE MA DA MODIFICARE
        if ep % 2 == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'loss': val_loss
            }, f'{save_dir}/checkpoint_epoch{ep}.pth')
            print(f' checkpoint completo salvato: checkpoint_epoch{ep}.pth')

    print('Miglior accuratezza validazione:', best_val_acc)
    writer.close() # [NUOVO]
    return model

