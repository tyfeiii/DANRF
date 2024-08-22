import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from prep import printProgressBar
# from networks.unet_parts import *
from networks.unet_model import UNet#这是学生的那个
from networks.GenNet import GenNet#这是合成的那个
from networks.models import UNet1 #这是预训练的那个
from measure import *
from loader import *
from Losses import *

class Solver(object):
    def __init__(self, args, data_loader):
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.save_path = args.save_path
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.patch_size = args.patch_size
        self.patch_n = args.patch_n

        self.UNet = UNet(1,1) #这是学生的那个
        self.GenNet = GenNet(1,1) #这是合成的那个
        self.teacher = UNet1(1,1) #这是预训练的那个
        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.UNet = nn.DataParallel(self.UNet)
            self.GenNet = nn.DataParallel(self.GenNet)
            self.teacher = nn.DataParallel(self.teacher)
        self.UNet.to(self.device)
        self.GenNet.to(self.device)
        self.teacher.to(self.device)

        self.lr = args.lr
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.mseloss = nn.MSELoss()
        self.contentloss = content_loss()
        self.Floss = Frobenius_loss()
        self.style_loss = style_loss()

        self.optimizer_stu = optim.Adam(self.UNet.parameters(), self.lr)
        self.scheduler1 = lr_scheduler.StepLR(self.optimizer_stu, step_size=self.decay_iters, gamma=0.5)
        self.optimizer_gen = optim.Adam(self.GenNet.parameters(), self.lr)
        self.scheduler2 = lr_scheduler.StepLR(self.optimizer_gen, step_size=self.decay_iters, gamma=0.5)

        # ema
        self.ema = args.ema
        if self.ema:
            self.ema_iters = args.ema_iters
            self.ema_part_parameters = args.ema_part_parameters
            self.ema_alpha = args.ema_alpha
            self.ema_decay = args.ema_decay

        #generate_picture
        self.gen_train = args.gen_train
        self.gen_epoch = args.gen_epoch
        self.supervised_label = args.supervised_label
        self.genpic_savepath = args.genpic_savepath
        self.save_genpic = args.save_genpic
        self.gen_x_path = self.genpic_savepath
        self.gen_y_path = self.genpic_savepath
        os.makedirs(self.gen_x_path, exist_ok=True)
        os.makedirs(self.gen_y_path, exist_ok=True)

    def update_ema_variables(self, model, ema_model, alpha, global_step, ema_part_parameters, ema_decay):
        # Use the true average until the exponential average is more correct
        if ema_decay:
            alpha = min(1 - 1 / (global_step + 1), alpha)
        else:
            pass

        with torch.no_grad():
            model_state_dict = model.state_dict()
            ema_model_state_dict = ema_model.state_dict()
            ema_list = list(ema_model_state_dict.keys())[ema_part_parameters[0]:ema_part_parameters[1]]
            for entry in ema_list:
                ema_param = ema_model_state_dict[entry].clone().detach()
                param = model_state_dict[entry].clone().detach()
                new_param = (ema_param * alpha) + (param * (1. - alpha))
                ema_model_state_dict[entry] = new_param
            ema_model.load_state_dict(ema_model_state_dict)

    def save_model(self, iter_):
        f = os.path.join(self.save_path, 'UNet_{}iter.ckpt'.format(iter_))
        # f1 = os.path.join(self.save_path, 'UNetteacher_{}iter.ckpt'.format(iter_))
        torch.save(self.UNet.state_dict(), f)
        # torch.save(self.teacher.state_dict(), f1)


    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'UNet_{}iter.ckpt'.format(iter_))
        # f = os.path.join(self.save_path, 'UNetteacher_{}iter.ckpt'.format(iter_))
        if self.multi_gpu:
            state_d = OrderedDict()
            for k, v in torch.load(f):
                n = k[7:]
                state_d[n] = v
            self.UNet.load_state_dict(state_d)
        else:
            self.UNet.load_state_dict(torch.load(f))


    # def lr_decay(self):
    #     lr = self.lr * 0.5
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.save_path, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def save_generate_image(self,x,x_path,y,y_path,idex):
        Xpic = x.data.cpu().numpy()
        Ypic = y.data.cpu().numpy()
        Xpicpath = os.path.join(x_path, "generate_input"+idex+".npy")
        Ypicpath = os.path.join(y_path, "generate_target"+idex+".npy")
        np.save(Xpicpath,Xpic)
        np.save(Ypicpath,Ypic)


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        # 加载teacher
        #####################################################
        f = os.path.join('/home/DANRF/networks/teacher-save/', 'UNet_{}iter.ckpt'.format(227900))#
        self.teacher.load_state_dict(torch.load(f))
        ########################################################


        for epoch in range(0, self.num_epochs):
            self.UNet.train(True)
            self.GenNet.train(True)
            if epoch >= self.gen_epoch and self.gen_train:
                renewdataset = ct_dataset(mode='train', load_mode=0, saved_path=self.genpic_savepath,
                                      test_patient='DL', patch_n=self.patch_n,
                                      patch_size=self.patch_size, transform=False,genlabel="generate")
                renewdata_loader = DataLoader(dataset=renewdataset, batch_size=1, shuffle=True, num_workers=4)

                data_loader_renew = iter(enumerate(renewdata_loader))
                data_loader_raw = iter(enumerate(self.data_loader))

                for i in range(len(self.data_loader)):
                    iter_,(x,y) = next(data_loader_raw)
                    total_iters += 1
                    # add 1 channel
                    x = x.unsqueeze(0).float().to(self.device)
                    y = y.unsqueeze(0).float().to(self.device)

                    if self.patch_size:  # patch training
                        x = x.view(-1, 1, self.patch_size, self.patch_size)
                        y = y.view(-1, 1, self.patch_size, self.patch_size)

                    loss_dis = 0
                    # denoise
                    self.UNet.zero_grad()
                    self.optimizer_stu.zero_grad()
                    # distillation stage
                    with torch.no_grad():
                        (pred_teacher, pred_feature) = self.teacher(x)

                    # pred_student = self.UNet(x)
                    (pred_student, feature_fig) = self.UNet(x)

                    # loss_ = self.criterion(pred, y)
                    loss0 = self.criterion(pred_student, pred_teacher)  # dill loss
                    loss1 = self.Floss(feature_fig, pred_feature)  # dill feature loss
                    loss_stu = 1 * loss0 + 1 * loss1

                    # loss = loss_
                    # loss = loss1

                    loss_stu.backward()  # retain_graph=True
                    self.optimizer_stu.step()

                    _, (rex, rey) = next(data_loader_renew)

                    if rex.shape[0] != 20:
                        rex = rex.unsqueeze(0).float().to(self.device)
                        rey = rey.unsqueeze(0).float().to(self.device)

                        if self.patch_size:  # patch training
                            rex = rex.view(-1, 1, self.patch_size, self.patch_size)
                            rey = rey.view(-1, 1, self.patch_size, self.patch_size)

                    with torch.no_grad():
                        (pred_teacher, pred_feature) = self.teacher(rex)

                    # pred_student = self.UNet(x)
                    (pred_student, feature_fig) = self.UNet(rex)
                    # loss_ = self.criterion(pred, y)
                    loss0 = self.criterion(pred_student, pred_teacher)  # dill loss
                    # loss1 = self.mseloss(feature_fig,pred_feature)#dill feature loss
                    # losst = self.Floss(feature_fig,pred_feature)
                    loss1 = self.Floss(feature_fig, pred_feature)  # dill feature loss
                    if self.supervised_label:
                        loss_supvise = self.criterion(pred_student,rey) #only l1 loss?
                        loss_stu = 1 * loss0 + 1 * loss1 + 0.1 * loss_supvise #weight is important
                    else:
                        loss_stu = 1 * loss0 + 1 * loss1

                    # loss = loss_
                    # loss = loss1

                    loss_stu.backward()  # retain_graph=True
                    self.optimizer_stu.step()
                    loss_dis += loss_stu.item()
                    loss_dis = loss_dis
                    # generate
                    self.GenNet.zero_grad()
                    self.optimizer_gen.zero_grad()
                    gen_fig = self.GenNet(x)  # generated fig

                    if self.save_genpic:
                        self.save_generate_image(gen_fig, self.gen_x_path, pred_student, self.gen_y_path, str(iter_))

                    # (gen_pred, gen_feature) = self.UNet(gen_fig)  # get feature of generated fig
                    # loss2 = self.mseloss(gen_feature, feature_fig.detach())  # style loss
                    loss2 = self.style_loss(gen_fig,x)
                    loss3 = self.contentloss(gen_fig, x)  # content loss
                    loss_gen = 1 * loss2 + 1 * loss3
                    # torch.autograd.set_detect_anomaly(True)
                    loss_gen.backward()  # retain_graph=True
                    self.optimizer_gen.step()

                    train_losses.append([loss_dis, loss_gen.item()])

                    # ema
                    if self.ema and total_iters % self.ema_iters == 0 :  #and total_iters>= 10 * self.save_iters
                        self.update_ema_variables(self.UNet, self.teacher, self.ema_alpha, total_iters,
                                                  self.ema_part_parameters, self.ema_decay)
                        # a problem is that should we replace the total_iters with epochs?
                        print("ema successful")
                    # print
                    if total_iters % self.print_iters == 0:
                        print(
                            "STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nlr: {:.20f}, StuTotalLOSS: {:.8f}, GenTotalLOSS: {:.8f},"
                            "L1LOSS: {:.8f},FeatureLOSS: {:.8f},\nStyleLOSS: {:.8f},ContentLOSS: {:.8f},"
                            "TIME: {:.1f}s".format(total_iters, epoch,
                                                   self.num_epochs, iter_ + 1,
                                                   len(self.data_loader),
                                                   self.optimizer_stu.state_dict()['param_groups'][0]['lr'],
                                                   loss_stu.item(), loss_gen.item(),
                                                   loss0.item(), loss1.item(),loss2.item(), loss3,
                                                   time.time() - start_time))
                    # learning rate decay
                    self.scheduler1.step()
                    self.scheduler2.step()
                    # save model
                    if total_iters % self.save_iters == 0:
                        self.save_model(total_iters)
                        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)),
                                np.array(train_losses))


            else:
                for iter_, (x, y) in enumerate(self.data_loader):
                    total_iters += 1

                    # add 1 channel
                    x = x.unsqueeze(0).float().to(self.device)
                    y = y.unsqueeze(0).float().to(self.device)

                    if self.patch_size: # patch training
                        x = x.view(-1, 1, self.patch_size, self.patch_size)
                        y = y.view(-1, 1, self.patch_size, self.patch_size)

                    #denoise
                    self.UNet.zero_grad()
                    self.optimizer_stu.zero_grad()

                    # distillation stage
                    with torch.no_grad():
                        (pred_teacher,pred_feature) = self.teacher(x)

                    # pred_student = self.UNet(x)
                    (pred_student,feature_fig)=self.UNet(x)

                    #loss_ = self.criterion(pred, y)
                    loss0 = self.criterion(pred_student, pred_teacher)# dill loss
                    loss1 = self.Floss(feature_fig,pred_feature)#dill feature loss
                    loss_stu = 1*loss0+ 1*loss1

                    # loss = loss_
                    # loss = loss1

                    loss_stu.backward() # retain_graph=True
                    self.optimizer_stu.step()


                    #generate
                    self.GenNet.zero_grad()
                    self.optimizer_gen.zero_grad()
                    gen_fig = self.GenNet(x)# generated fig

                    if self.save_genpic and epoch == self.gen_epoch-1:
                        self.save_generate_image(gen_fig,self.gen_x_path,pred_student,self.gen_y_path,str(iter_))

                    # (gen_pred,gen_feature) = self.UNet(gen_fig) #get feature of generated fig
                    # loss2 = self.mseloss(gen_feature,feature_fig.detach())#style loss
                    loss2 = self.style_loss(gen_fig,x)
                    loss3 = self.contentloss(gen_fig,x)#content loss
                    loss_gen = 1*loss2 +1*loss3
                    # torch.autograd.set_detect_anomaly(True)
                    loss_gen.backward() # retain_graph=True
                    self.optimizer_gen.step()

                    train_losses.append([loss_stu.item(),loss_gen.item()])

                    # ema
                    if self.ema and total_iters%self.ema_iters == 0  : #and total_iters>= 10 * self.save_iters
                        self.update_ema_variables(self.UNet, self.teacher, self.ema_alpha, total_iters, self.ema_part_parameters, self.ema_decay)
                        # a problem is that should we replace the total_iters with epochs?
                        print("ema successful")
                    # print
                    if total_iters % self.print_iters == 0:
                        print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nlr: {:.20f}, StuTotalLOSS: {:.8f}, GenTotalLOSS: {:.8f},"
                              "L1LOSS: {:.8f},FeatureLOSS: {:.8f},\nStyleLOSS: {:.8f},ContentLOSS: {:.8f},"
                              "TIME: {:.1f}s".format(total_iters, epoch,
                              self.num_epochs, iter_+1,
                                                     len(self.data_loader), self.optimizer_stu.state_dict()['param_groups'][0]['lr'],loss_stu.item(),loss_gen.item(),
                                                     loss0.item(),loss1.item(),loss2.item(),loss3,
                                                     time.time() - start_time))
                    # learning rate decay
                    self.scheduler1.step()
                    self.scheduler2.step()
                    # save model
                    if total_iters % self.save_iters == 0:
                        self.save_model(total_iters)
                        np.save(os.path.join(self.save_path, 'loss_{}_iter.npy'.format(total_iters)), np.array(train_losses))
            # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????

            if total_iters % self.save_iters == 0 and total_iters != 0:    # 修改
                dataset_ = ct_dataset(mode='test', load_mode=0, saved_path='/data/tangyufei/dataset/axis/val2',
                                      test_patient='LDCT', patch_n=None,
                                      patch_size=None, transform=False)
                data_loader = DataLoader(dataset=dataset_, batch_size=None, shuffle=True, num_workers=0)


                with torch.no_grad():

                    for i, (x, y) in enumerate(data_loader):

                        shape_ = x.shape[-1]
                        x = x.unsqueeze(0).float().to(self.device)
                        y = y.unsqueeze(0).float().to(self.device)
                        x = x.unsqueeze(0).float().to(self.device)
                        y = y.unsqueeze(0).float().to(self.device)
                        # pred = self.WGAN_VGG_generator1(x)
                        (pred,feature) = self.UNet(x)
                        x1 = self.trunc(self.denormalize_(x))
                        y1 = self.trunc(self.denormalize_(y))
                        pred1 = self.trunc(self.denormalize_(pred))
                        data_range = self.trunc_max - self.trunc_min
                        original_result, pred_result = compute_measure(x1, y1, pred1, data_range)
                        pred_psnr_avg += pred_result[0]
                        pred_ssim_avg += pred_result[1]
                        pred_rmse_avg += pred_result[2]



            #########################################################
            # 日志文件

                with open('./save1/pred_psnr_avg.txt', 'a') as f:
                    f.write('EPOCH:%d loss:%.20f' % (epoch, pred_psnr_avg / len(data_loader)) + '\n')
                    f.close()

                with open('./save1/pred_ssim_avg.txt', 'a') as f:
                    f.write('EPOCH:%d loss:%.20f' % (epoch, pred_ssim_avg / len(data_loader)) + '\n')
                    f.close()

                with open('./save1/pred_rmse_avg.txt', 'a') as f:
                    f.write('EPOCH:%d loss:%.20f' % (epoch, pred_rmse_avg / len(data_loader)) + '\n')
                    f.close()
                pred_psnr_avg = 0
                pred_ssim_avg = 0
                pred_rmse_avg = 0
            #########################################################
            # else:
            #     continue

    def test(self):
        del self.UNet
        # load
        # self.AENet = AENet(input_nc=1, output_nc=1).to(self.device)
        self.UNet = UNet(1,1).to(self.device)
        self.load_model(self.test_iters)

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        ori_psnr_avg1, ori_ssim_avg1, ori_rmse_avg1 = [], [], []
        pred_psnr_avg1, pred_ssim_avg1, pred_rmse_avg1 = [], [], []

        with torch.no_grad():
            for i, (x, y) in enumerate(self.data_loader):

                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)
                # real_noise = x-y
                (pred,feature) = self.UNet(x)

                # denormalize, truncate
                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                # real_noise = self.trunc(self.denormalize_(real_noise.view(shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                np.save(os.path.join(self.save_path, 'x', '{}_result'.format(i)), x)
                np.save(os.path.join(self.save_path, 'y', '{}_result'.format(i)), y)
                # np.save(os.path.join(self.save_path, 'real', '{}_result'.format(i)), real_noise)
                np.save(os.path.join(self.save_path, 'pred', '{}_result'.format(i)), pred)

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_psnr_avg1.append(ori_psnr_avg / len(self.data_loader))
                ori_ssim_avg += original_result[1]
                ori_ssim_avg1.append(ori_ssim_avg / len(self.data_loader))
                ori_rmse_avg += original_result[2]
                ori_rmse_avg1.append(ori_rmse_avg / len(self.data_loader))
                pred_psnr_avg += pred_result[0]
                pred_psnr_avg1.append(pred_psnr_avg / len(self.data_loader))
                pred_ssim_avg += pred_result[1]
                pred_ssim_avg1.append(pred_ssim_avg / len(self.data_loader))
                pred_rmse_avg += pred_result[2]
                pred_rmse_avg1.append(pred_rmse_avg / len(self.data_loader))

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, i, original_result, pred_result)


            print('\n')
            print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.data_loader), ori_ssim_avg / len(self.data_loader),
                ori_rmse_avg / len(self.data_loader)))
            print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.data_loader), pred_ssim_avg / len(self.data_loader),
                pred_rmse_avg / len(self.data_loader)))







            #     printProgressBar(i, len(self.data_loader),
            #                      prefix="Compute measurements ..",
            #                      suffix='Complete', length=25)
            # print('\n')
            # print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(ori_psnr_avg/len(self.data_loader),
            #                                                                                 ori_ssim_avg/len(self.data_loader),
            #                                                                                 ori_rmse_avg/len(self.data_loader)))
            # print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(pred_psnr_avg/len(self.data_loader),
            #                                                                                       pred_ssim_avg/len(self.data_loader),
            #                                                                                       pred_rmse_avg/len(self.data_loader)))
        #
        # '''PSNR,SSIM,RMSE IMAGES'''
        # fig = plt.figure()
        # plt.title('PSNR')
        # ax1 = fig.add_subplot(1, 1, 1)
        # ax1.plot(list(range(len(self.data_loader))), ori_psnr_avg1, label='ori_psnr')
        # ax1.plot(list(range(len(self.data_loader))), pred_psnr_avg1, label='pred_psnr')
        # plt.legend()  # 显示图例
        # plt.xlabel('image_num')
        # plt.ylabel('psnr')
        # plt.savefig('psnr.png')
        #
        # fig = plt.figure()
        # plt.title('SSIM')
        # ax2 = fig.add_subplot(1, 1, 1)
        # ax2.plot(list(range(len(self.data_loader))), ori_ssim_avg1, label='ori_ssim')
        # ax2.plot(list(range(len(self.data_loader))), pred_ssim_avg1, label='pred_ssim')
        # plt.legend()  # 显示图例
        # plt.xlabel('image_num')
        # plt.ylabel('ssim')
        # plt.savefig('ssim.png')
        #
        # fig = plt.figure()
        # plt.title('RMSE')
        # ax3 = fig.add_subplot(1, 1, 1)
        # ax3.plot(list(range(len(self.data_loader))), ori_rmse_avg1, label='ori_rmse')
        # ax3.plot(list(range(len(self.data_loader))), pred_rmse_avg1, label='pred_rmse')
        # plt.legend()  # 显示图例
        # plt.xlabel('image_num')
        # plt.ylabel('rmse')
        # plt.savefig('rmse.png')
        # plt.show()
