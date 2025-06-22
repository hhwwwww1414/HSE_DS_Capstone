# --- train_products.py -----------------------------------------------
import lightning as L, torch, torchvision as tv
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import (Compose, ToTensor, Normalize,
                                     RandomHorizontalFlip)
from pathlib import Path

def main():
    BATCH = 64
    train_tf = Compose([RandomHorizontalFlip(), ToTensor(),
                        Normalize([0.5]*3, [0.5]*3)])
    test_tf  = Compose([ToTensor(), Normalize([0.5]*3, [0.5]*3)])

    data = ImageFolder("data/interim/products_224", transform=train_tf)
    n_val = int(0.1*len(data)); n_train = len(data)-n_val
    train_ds, val_ds = random_split(data, [n_train, n_val])

    train_dl = DataLoader(train_ds, BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

    model = tv.models.mobilenet_v3_small(num_classes=len(data.classes))
    loss_fn = torch.nn.CrossEntropyLoss()

    class Lit(L.LightningModule):
        def __init__(self, net): super().__init__(); self.net = net
        def forward(self,x): return self.net(x)
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=2e-4)
        def _step(self,b,stage):
            x,y=b; yhat=self(x)
            loss=loss_fn(yhat,y)
            acc=(yhat.argmax(1)==y).float().mean()
            self.log_dict({f"{stage}_loss":loss, f"{stage}_acc":acc}, prog_bar=True)
            return loss
        def training_step(self,b,i):  return self._step(b,"train")
        def validation_step(self,b,i):return self._step(b,"val")

    trainer = L.Trainer(max_epochs=8, accelerator="auto")
    trainer.fit(Lit(model), train_dl, val_dl)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint("data/models/mobilenet_v3_small.ckpt")

if __name__ == "__main__":            # ← главный блок
    main()
# ----------------------------------------------------------------------
