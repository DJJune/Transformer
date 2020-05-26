
from network import Transformer
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
import torch as t
import torch.nn as nn
from tqdm import tqdm
import hyperparameters as hp

# warm up the model


def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * \
        min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():

    global_step = 0

    # multi GPUs
    m = nn.DataParallel(Transformer().cuda())

    # apply batchNorm and Dropout
    # if using m.eval(), the batch normaliaztion and droput calculation would be unavailable
    m.train()

    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    # use tensorboard to record the training information
    writer = SummaryWriter()

    for epoch in tqdm(range(hp.epochs)):

        """
        # if  use dataloader to collect the training sample
        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
        for i, data in enumerate(dataloader):
            pbar.set_description("Processing at epoch %d"%epoch)
        """

        # I use a simple example here
        for itera in range(10):

            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)

            # how are you eos ->
            # what is your name eos ->
            source_seq = t.Tensor([[2, 3, 4, 1, 0],
                                   [5, 6, 7, 8, 1]]).long()

            # -> I am fine , thanks eos
            # -> Bob eos
            target_seq = t.Tensor([[9, 10, 11, 12, 13, 1],
                                   [14, 1,  0,  0,  0, 0]]).long()

            # right-shifted target sequence
            target_seq_input = t.Tensor([[0,  9, 10, 11, 12, 13],
                                         [0, 14,  1,  0,  0,  0]]).long()

            source_pos = t.Tensor([[1, 2, 3, 4, 0],
                                   [1, 2, 3, 4, 5]]).long()
            target_pos = t.Tensor([[1, 2, 3, 4, 5, 6],
                                   [1, 2, 0, 0, 0, 0]]).long()

            source_seq = source_seq.cuda()
            target_seq = target_seq.cuda()
            target_seq_input = target_seq_input.cuda()
            source_pos = source_pos.cuda()
            target_pos = target_pos.cuda()

            pred_logit, attn_probs, attns_enc, attns_dec = m.forward(source_seq, target_seq_input,
                                                                     source_pos, target_pos)

            # reshape the pred tensor to (B, T * vocab_size)
            loss = nn.CrossEntropyLoss()(
                pred_logit.reshape(-1, pred_logit.size()[2]), target_seq.reshape(-1))

            writer.add_scalars('training_loss', {
                'loss': loss,
            }, global_step)

            """
            # You can draw the attention map here

            if global_step % hp.image_step == 1:
                
                for i, prob in enumerate(attn_probs):
                    
                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_%d_0'%global_step, x, i*4+j)
                
                for i, prob in enumerate(attns_enc):
                    num_h = prob.size(0)
                    
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_enc_%d_0'%global_step, x, i*4+j)
            
                for i, prob in enumerate(attns_dec):

                    num_h = prob.size(0)
                    for j in range(4):
                
                        x = vutils.make_grid(prob[j*16] * 255)
                        writer.add_image('Attention_dec_%d_0'%global_step, x, i*4+j)
            """

            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()

            nn.utils.clip_grad_norm_(m.parameters(), 1.)

            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                if not os.path.exists(hp.checkpoint_path):
                    os.mkdir(hp.checkpoint_path)
                t.save({'model': m.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(hp.checkpoint_path, 'checkpoint_transformer_%d.pth.tar' % global_step))


if __name__ == '__main__':
    main()
