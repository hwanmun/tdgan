import numpy as np
import torch
from tqdm import tqdm
from torchsummary import summary
from losses import gen_BCE_logit_loss, disc_BCE_logit_loss
from evaluation import EvaluatorByTraining

class TrainingInterfaceGAN:
    """ Interface class to train GAN model. If model_path specified, the final
    generator model will be stored in that path. If log_path specified, losses
    of generator and discriminator models are stored in a log file, as well as
    the evaluation result at end of each epoch when eval_kwargs specified.
    """
    def __init__(self, generator, discriminator, train_set, dev_set,
                 latent_dim, generator_optimizer_class=torch.optim.Adam,
                 discriminator_optimizer_class=torch.optim.Adam,
                 generator_loss_fn=gen_BCE_logit_loss,
                 discriminator_loss_fn=disc_BCE_logit_loss,
                 device='cpu', model_path=None, log_path=None,
                 eval_kwargs=None):
        self.device = device
        self.gen = generator.to(self.device)
        self.disc = discriminator.to(self.device)
        self.gen_optim_cls = generator_optimizer_class
        self.disc_optim_cls = discriminator_optimizer_class
        self.gen_optim = None
        self.disc_optim = None
        self.gen_loss = generator_loss_fn
        self.disc_loss = discriminator_loss_fn
        self.latent_dim = latent_dim
        self.train_set = train_set
        self.dev_set = dev_set
        self.model_path = model_path
        self.log_path = log_path
        self.evaluator = None
        if eval_kwargs is not None:
            self.evaluator = EvaluatorByTraining(self.dev_set, **eval_kwargs)

        self.training_log = {}

        print('Generator:')
        summary(self.gen, (self.latent_dim,), device=self.device)
        print('Discriminator:')
        summary(self.disc, (self.train_set.feat_dim,), device=self.device)
    
    def train(self, epochs, batch_size, gen_opt_kwargs={}, disc_opt_kwargs={},
              running_avg_size=100, train_tracking_period=10000, verbose=True):
        self.gen_optim = self.gen_optim_cls(self.gen.parameters(),
                                            **gen_opt_kwargs)
        self.disc_optim = self.disc_optim_cls(self.disc.parameters(),
                                              **disc_opt_kwargs)

        train_dataloader = self.train_set.get_dataloader(
            batch_size=batch_size, shuffle=True, drop_last=True,
            collate_fn='noise'
        )

        self.training_log = {'losses': []}
        if self.evaluator is not None:
            self.training_log['eval_metric'] = []
        self.gen.train()
        self.disc.train()
        gen_batch_losses, disc_batch_losses, step_count = [], [], 0
        for epoch in range(epochs):
            if verbose:
                print(f'starting training epoch {epoch + 1}:')
            for batch in tqdm(train_dataloader):
                batch = batch.to(self.device)

                self.disc_optim.zero_grad()
                z = torch.randn(batch.shape[0], self.latent_dim,
                                device=self.device)
                fake_data = self.gen(z)
                real_pred = self.disc(batch)
                fake_pred = self.disc(fake_data.detach())

                disc_loss = self.disc_loss(real_pred, fake_pred)
                disc_batch_losses.append(disc_loss.item())
                disc_loss.backward()
                self.disc_optim.step()

                self.gen_optim.zero_grad()
                z = torch.randn(batch.shape[0], self.latent_dim,
                                device=self.device)
                fake_data = self.gen(z)
                fake_pred = self.disc(fake_data)
                gen_loss = self.gen_loss(fake_pred)
                gen_batch_losses.append(gen_loss.item())
                gen_loss.backward()
                self.gen_optim.step()

                step_count += 1
                if len(disc_batch_losses) > running_avg_size:
                    disc_batch_losses.pop(0)
                if len(gen_batch_losses) > running_avg_size:
                    gen_batch_losses.pop(0)
                if step_count % train_tracking_period == 0:
                    gen_loss = np.mean(gen_batch_losses)
                    disc_loss = np.mean(disc_batch_losses)
                    self.training_log['losses'].append(
                        (step_count, gen_loss, disc_loss)
                    )
                    if verbose:
                        print(f'... step {step_count}, '
                              f'gen_loss={gen_loss}, disc_loss={disc_loss}')
            
            self.epoch_ending()
        
        if self.model_path is not None:
            state = {
                'state_dict': self.gen.state_dict(),
                'key_to_idx': self.train_set.key_to_idx,
                'ds_stats': self.train_set.stats,
                'latent_dim': self.latent_dim
            }
            torch.save(state, self.model_path)
        
        if self.log_path is not None:
            with open(self.log_path, 'w') as f:
                for key, vals in self.training_log.items():
                    f.write(key+'\n')
                    for val in vals:
                        f.write(','.join([str(v) for v in list(val)]) + '\n')
    
    def epoch_ending(self):
        self.gen.eval()
        if self.evaluator is not None:
            val, (f, r) = self.evaluator.evaluate(self.gen, self.latent_dim)
            self.training_log['eval_metric'].append((val, f, r))
            print(f'eval={val}, fake_metric={f}, real_metric={r}')
        self.gen.train()
                    
                





    
