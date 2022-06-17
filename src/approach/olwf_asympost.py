import torch
import math
from copy import deepcopy
from argparse import ArgumentParser
import torch.nn.functional as F

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset
from networks.early_conv_vit_net import start_post_rec, stop_post_rec, get_post_attention_list 
from einops import rearrange#, reduce, repeat


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, sparsefact=100.,  plast_mu=1, lamb=1, T=2, sym=False, use_pod_factor=False, pool_along='spatial'):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.model_old = None
        self.sparsefact = sparsefact
        self.lamb = lamb
        self.T = T
        self.plast_mu = plast_mu
        self._task_size = 0
        self._n_classes = 0
        self._pod_spatial_factor = 3.
        self.sym = sym
        self.use_pod_factor = use_pod_factor
        self.pool_along = pool_along

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--sparsefact', default=100., type=float, required=False, help='add sparse attention regularization for asym loss')
        parser.add_argument('--sym', action='store_true', default=False, required=False,
                            help='Use symmetric version of the loss if given (default=%(default)s)')
        parser.add_argument('--plast_mu', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument('--T', default=2, type=int, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--use-pod-factor', action='store_true', default=False, required=False,
                            help='Use pod factor to weigh sym/asym losses if given (default=%(default)s)')
        parser.add_argument('--pool-along', default='spatial', required=False)
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def plasticity_loss(self, old_attention_list, attention_list):
        """ jensen shannon (JS) plasticity loss between the attention maps
            of the old model and the new model, we sum the mean JS for each layer.
            Tiny ViTs models have 12 layers: each layer has 3 heads, the attention map size is (197,197).
            you will have a len(attention_list) = 12
            and each element of the list is (batch_size,3,197,197)
            we compute the JS on the columns, after normalizing (transforming the columns in probabilities).
        """

        totloss = 0.
        for i in range(len(attention_list)):

            # reshape
            p = rearrange(old_attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')
            q = rearrange(attention_list[i].view(-1, 197,197).to(self.device), 'b h w -> (b w) h')

            # get rid of negative values
            p = torch.abs(p)
            q = torch.abs(q)

            # transform them in probabilities
            p /=  p.sum(dim=1).unsqueeze(1)
            q /= q.sum(dim=1).unsqueeze(1)

            # JS
            m = (1./2.) * (p + q)
            t1 = (1./2.) * (p * ((p / m)+1e-05).log()).sum(dim=1)
            t2 = (1./2.) * (q * ((q / m)+1e-05).log()).sum(dim=1)
            loss = t1 + t2

            # we sum the mean for each layer
            totloss += loss.mean()

        return totloss
    
    def permissive_relu(self, att_diff, asym_choice):
        relu_out_ = asym_choice(att_diff)
        penalty_factor = math.log(math.sqrt(
                    self._n_classes / self._task_size
                ))
        scaled_att_diff = torch.abs(att_diff) * penalty_factor
        # scaled_att_diff = torch.abs(att_diff) / 2.0 # make the negative values go smaller after abs() so that they are penalized less
        zero_relu_indices = relu_out_ == 0
        relu_out = relu_out_.clone()
        relu_out[zero_relu_indices] = scaled_att_diff[zero_relu_indices]
        return relu_out

 
    def pod(self,
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True,
        asymmetric=False
    ):
        """Pooled Output Distillation.
        Reference:
            * Douillard et al.
            Small Task Incremental Learning.
            arXiv 2020.
        :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
        :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
        :param collapse_channels: How to pool the channels.
        :param memory_flags: Integer flags denoting exemplars.
        :param only_old: Only apply loss to exemplars.
        :return: A float scalar loss.
        """

        #import pdb; pdb.set_trace()
        loss = torch.tensor(0.).to(self.device)
        layers_to_pool = range(len(list_attentions_a))
        # for i in layers_to_pool:
            # shape of (b, n, w, h)
            # p = rearrange(list_attentions_a[i].to(self.device), 's h b w -> h s b w') # rearrange to make head as the first dimension
            # q = rearrange(list_attentions_b[i].to(self.device), 's h b w -> h s b w')

        # list_attentions_a[1].shape = (1024, 12, 5, 64) ==>> 64 is the spatiality
        for idx, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
            assert a.shape == b.shape, (a.shape, b.shape)
            if not asymmetric:
                a = torch.pow(a, 2)
                b = torch.pow(b, 2)

            #collapse_channels = 'channels' # collassa 197
            if collapse_channels == "spatial":
                a_h = a.sum(dim=2).view(a.shape[0], -1)  # [bs, w]
                b_h = b.sum(dim=2).view(b.shape[0], -1)  # [bs, w]
                a_w = a.sum(dim=3).view(a.shape[0], -1)  # [bs, h]
                b_w = b.sum(dim=3).view(b.shape[0], -1)  # [bs, h]
                a = torch.cat([a_h, a_w], dim=-1) # concatenates two [96, 197] to give [96, 394], dim = -1 does concatenation along the last axis
                b = torch.cat([b_h, b_w], dim=-1)
            elif collapse_channels == "gap":
                # compute avg pool2d over each 32x32 image to reduce the dimension to 1x1
                a = F.adaptive_avg_pool2d(a, (1, 1))[..., 0, 0] # [..., 0, 0] preserves only the [0][0]th element of last two dimensions, i.e., [96, 197, 197] into [96], since 197x197 reduced to 1x1 and pooled together
                b = F.adaptive_avg_pool2d(b, (1, 1))[..., 0, 0]
            elif collapse_channels == "width":
                a = a.sum(dim=3).view(a.shape[0], -1)  # shape of (b, c * h)
                b = b.sum(dim=3).view(b.shape[0], -1)
            elif collapse_channels == "height":
                a = a.sum(dim=2).view(a.shape[0], -1)  # shape of (b, c * w)
                b = b.sum(dim=2).view(b.shape[0], -1)
            elif collapse_channels == "pixel":
                pass
            else:
                raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

            distance_loss_weight = self.pod_spatial_factor if self.use_pod_factor else self.plast_mu
            if asymmetric:
                asym_choice = torch.nn.ReLU(inplace=True)
                if normalize:
                    a = F.normalize(a, dim=1, p=2)
                    b = F.normalize(b, dim=1, p=2)
                diff = a-b
                relu_out = asym_choice(diff)  
                layer_loss = torch.mean(torch.frobenius_norm(relu_out, dim=-1)) * distance_loss_weight
            else:
                if normalize:
                    a = F.normalize(a, dim=1, p=2)
                    b = F.normalize(b, dim=1, p=2)
                layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1)) * distance_loss_weight
            loss += layer_loss 
        # loss = loss / len(p)

        return loss / len(layers_to_pool)
    

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for i, (images, targets) in enumerate(trn_loader):
            loss = 0.
            plastic_loss = 0.

            # Forward old model
            targets_old = None
            if t > 0:
                self.model_old.to(self.device)
                start_post_rec()
                targets_old = self.model_old(images.to(self.device))
                stop_post_rec()

                old_postattention_list = get_post_attention_list()

            # Forward current model
            start_post_rec()
            outputs = self.model(images.to(self.device))
            stop_post_rec()

            postattention_list = get_post_attention_list()

            self.pod_spatial_factor = self._pod_spatial_factor * math.sqrt(
                    self._n_classes / self._task_size
                )

            if t > 0:
                # plastic_loss += self.plasticity_loss(old_attention_list, attention_list)*self.plast_mu
                """ Headwise asymmetric loss (4 possible settings):
                for symmetric version of the loss, set asymmetric_loss = False
                for simple asym version, set asymmetric_loss = True , i.e., sparse_reg = None by default)
                for asym version with sparse attention with mean of |b|, set sparse_reg = 'mean'
                for asym version with sparse attention with norm of |b|, set sparse_reg = 'norm'
                """
                asym = True if not self.sym else False
                plastic_loss += self.pod(old_postattention_list, postattention_list, asymmetric=asym, collapse_channels=self.pool_along)

            loss += self.criterion(t, outputs, targets.to(self.device), targets_old)
            print(f"[Task {t}] l:{loss:.3f} p:{plastic_loss:.3f}")
            loss += plastic_loss


            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()



    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images.to(self.device))
                # Forward current model
                outputs = self.model(images.to(self.device))
                loss = self.criterion(t, outputs, targets.to(self.device), targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""
        loss = 0
        if t > 0:
            # Knowledge distillation loss for all previous tasks
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)
        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            return loss + torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

