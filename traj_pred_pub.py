""" chunk data in T_total length with data_id """

import pandas as pd
import sqlite3
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.distributions.normal import Normal

from PIL import Image

import math
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import datetime
import cv2
from skimage.util import img_as_ubyte
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import argparse

# Log
from torch.utils.tensorboard import SummaryWriter

stochastic_mode         = 1 #output

# Make folder for outputs and logs
parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", default="zara_01", choices=["hotel","eth","zara_01","zara_02","university"])
args = parser.parse_args()

dataset_name = args.dataset  # dataset options: 'university', 'zara_01', 'zara_02', 'eth', 'hotel'
run_folder  = "**update path**/traj_pred_"+ dataset_name +"_" + str(os.path.basename(__file__))+ '_' + str( datetime.datetime.now() ) 
os.makedirs(run_folder)   

# Make log folder for tensorboard
SummaryWriter_path = run_folder + "/log"
os.makedirs(SummaryWriter_path)   
writer = SummaryWriter(SummaryWriter_path,comment="ADE_FDE_Train")

# Make image folder to save outputs
image_path  = run_folder + "/Visual_Prediction"
os.makedirs(image_path)

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DataBase Variables
image_folder_path       = '**update path**/data_trajpred/'+dataset_name
DB_PATH_train     = "**update path**/data_trajpred/"+dataset_name+"/pos_data_train.db"
cnx_train         = sqlite3.connect(DB_PATH_train)
DB_PATH_val     = "**update path**/data_trajpred/"+dataset_name+"/pos_data_val.db"
cnx_val         = sqlite3.connect(DB_PATH_val)
DB_DIR      = run_folder + '/database'
os.makedirs( DB_DIR )
DB_PATH2    = DB_DIR+'/db_one_ped_delta_coordinates_results.db'
cnx2        = sqlite3.connect(DB_PATH2)

# Variables
T_obs                   = 8
T_pred                  = 12
T_total                 = T_obs + T_pred
data_id                 = 0 
batch_size              = 100#15
chunk_size              = batch_size * T_total # Chunksize should be multiple of T_total
in_size                 = 2
stochastic_out_size     = in_size * 2
hidden_size             = 256 #!64
embed_size              = 64 #16 #!64
global dropout_val
dropout_val             = 0.2 #0.5
teacher_forcing_ratio   = 0.7 # 0.9
regularization_factor   = 0.5 # 0.001
avg_n_path_eval         = 20
bst_n_path_eval         = 20
path_mode               = "top5" #"avg","bst","single","top5"
regularization_mode     = "regular" #"weighted","e_weighted", "regular"
startpoint_mode         = "on" #"on","off"
enc_out                 = "on" #"on","off"
biased_loss_mode        = 0 # 0 , 1

table_out   = "results_delta"
table       = "dataset_T_length_"+str(T_total)+"delta_coordinates"
df_id       = pd.read_sql_query("SELECT data_id FROM "+table, cnx_train)
data_size   = df_id.data_id.max() * T_total
epoch_num   = 200 #200
from_epoch  = 0

# Visual Variables
image_size              = 256  
image_dimension         = 3
mask_size               = 16
visual_features_size    = 128 
visual_embed_size       = 64  #128 #256 #64
vsn_module_out_size    = 256
to_pil = torchvision.transforms.ToPILImage()
# vgg16_intermediate_size = 7*7*512

# Model Path
model_path = run_folder + "/NNmodel" 
os.makedirs(model_path)   
model_path = model_path + str("/model")

if dataset_name == 'eth' or dataset_name =='hotel':   # ETH dataset
    h = np.array([[0.0110482,0.000669589,-3.32953],[-0.0015966,0.0116324,-5.39514],[0.000111907,0.0000136174,0.542766]])
else:                                       # UCY dataset
    h = np.array([[47.51,0,476],[0,41.9,117],[0,0,1]])

# ------------------------------------------------------------------------------
# Handle Sequential Data
class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, ROOT_DIR, DB_PATH,cnx):
        
        self.pos_df    = pd.read_sql_query("SELECT * FROM "+str(table), cnx)
        self.root_dir  = ROOT_DIR+'/visual_data'
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((image_size,image_size)), \
                                                         torchvision.transforms.ToTensor(), \
                                                         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.visual_data = []
        # read sorted frames
        for img in sorted(os.listdir(self.root_dir)): 
            self.visual_data.append(self.transform( Image.open(os.path.join(self.root_dir)+"/"+img) ))
        self.visual_data = torch.stack(self.visual_data)
       
    
    def __len__(self):
        return self.pos_df.data_id.max()
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        extracted_df     = self.pos_df[ self.pos_df["data_id"] == idx ]
        
        tensor           = torch.tensor(extracted_df[['pos_x_delta','pos_y_delta']].values).reshape(-1,T_total,in_size)
        obs, pred        = torch.split(tensor,[T_obs,T_pred],dim=1)
        
        start_frames     = (extracted_df.groupby('data_id').frame_num.min().values/10).astype('int')
        extracted_frames = []
        for i in start_frames:            
            extracted_frames.append(self.visual_data[i:i+T_obs])
        frames = torch.stack(extracted_frames)
        start_frames = torch.tensor(start_frames)
        return obs, pred, frames, start_frames

#Inverse Transform

inverse_transform = torchvision.transforms.Compose([ 
                                        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),\
                                        torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),\
                                        ])

# ------------------------------------------------------------------------------
# Initialize random weights for NN models
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.2, 0.2)


# ------------------------------------------------------------------------------
# Regularizer loss
sum_sigma_distance  = torch.zeros(1)

def distance_from_line_regularizer(input_tensor,prediction):
    global sum_sigma_distance
    # Fit a line to observation points over batch 
    input_tensor    = input_tensor.double()
    prediction      = prediction.double()
    input_tensor    = input_tensor.cumsum(dim=1).double()
    X               = torch.ones_like(input_tensor).double().to('cuda', non_blocking=True)
    X[:,:,0]        = input_tensor[:,:,0]
    Y               = (input_tensor[:,:,1]).unsqueeze(-1).double()
    try:
        try:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().inverse()
        except:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().pinverse()
        XTY             = torch.matmul( X.transpose(-1,-2), Y)
        theta           = torch.matmul( XTX_1.double(), XTY.double())
        # Calculate real values of prediction instead of delta
        prediction[:,:,0] = prediction[:,:,0] + input_tensor[:,-1,0].unsqueeze(-1) 
        prediction[:,:,1] = prediction[:,:,1] + input_tensor[:,-1,1].unsqueeze(-1)
        
        # Calculate distance ( predicted_points , observation_fitted_line ) over batch
        theta0x0        = theta[:,0,:].double() * prediction[:,:,0].double()
        denominator     = torch.sqrt( theta[:,0,:].double() * theta[:,0,:].double() + 1 )
        nominator       = theta0x0 + theta[:,1,:] - prediction[:,:,1].double()
        distance        = nominator.abs() / denominator
        if regularization_mode =='weighted':
            weight              = torch.flip( torch.arange(1,T_pred+1).cuda().float(),[0])
            weight              = (weight / T_pred).repeat(distance.size(0)).view(-1,T_pred)
            weighted_distance   = weight * distance

        elif regularization_mode =='e_weighted':
            weight              = torch.flip( torch.arange(1,T_pred+1).cuda().float(),[0])
            weight              = (weight / T_pred).repeat(distance.size(0)).view(distance.size(0),T_pred)
            weight              = torch.exp(weight)
            weighted_distance   = weight*distance

        else:
            weighted_distance = distance
        sigma_distance  = torch.mean(weighted_distance,1)
        sum_sigma_distance  = torch.mean(sigma_distance)
        return sum_sigma_distance
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1).to('cuda', non_blocking=True) + 20
        return sum_sigma_distance



def angle_from_line_regularizer(input_tensor,prediction):
    global sum_sigma_distance
    input_tensor    = input_tensor.double()
    prediction      = prediction.double()

    # Calculate real values of observation instead of delta
    input_tensor    = input_tensor.cumsum(dim=1).double()

    # Calculate real values of prediction instead of delta
    prediction[:,:,0] = prediction[:,:,0] + input_tensor[:,-1,0].unsqueeze(-1) 
    prediction[:,:,1] = prediction[:,:,1] + input_tensor[:,-1,1].unsqueeze(-1)

    # Fit a line to observation points over batch 
    X               = torch.ones_like(input_tensor).double()
    X[:,:,0]        = input_tensor[:,:,0]
    Y               = (input_tensor[:,:,1]).unsqueeze(-1).double()
    try:
        try:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().inverse()
        except:
            XTX_1           = torch.matmul( X.transpose(-1,-2), X).double().pinverse()
        XTY             = torch.matmul( X.transpose(-1,-2), Y)
        theta           = torch.matmul( XTX_1.double(), XTY.double())
        theta_observation   = theta.double()
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1) + math.pi/4
        return sum_sigma_distance  

    # Fit a line to prediction points over batch 
    X               = torch.ones_like(prediction).double()
    X[:,:,0]        = prediction[:,:,0]
    Y               = (prediction[:,:,1]).unsqueeze(-1).double()
    try:
        XTX_1           = torch.matmul( X.transpose(-1,-2), X).inverse()
        XTY             = torch.matmul( X.transpose(-1,-2), Y)
        theta           = torch.matmul( XTX_1.double(), XTY.double())
        theta_prediction    = theta.double()
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1) + math.pi/4
        return sum_sigma_distance 

    try:
        # Find two vectors(directed lines)
        x_first             = input_tensor[:,0,0].unsqueeze(-1)
        x_last              = input_tensor[:,-1,0].unsqueeze(-1)
        y_first             = theta_observation[:,0,:].double() * x_first.double()  +  theta_observation[:,1,:].double()
        y_last              = theta_observation[:,0,:].double() * x_last.double()  +  theta_observation[:,1,:].double()
        vector_observation  = [x_last-x_first , y_last-y_first]

        x_first             = prediction[:,0,0].unsqueeze(-1)
        x_last              = prediction[:,-1,0].unsqueeze(-1)
        y_first             = theta_prediction[:,0,:].double() * x_first.double()  +  theta_prediction[:,1,:].double()
        y_last              = theta_prediction[:,0,:].double() * x_last.double()  +  theta_prediction[:,1,:].double()
        vector_prediction   = [x_last-x_first , y_last-y_first]

        # Find the angle between two vectors
        nominator                   = vector_observation[0]*vector_prediction[0] + vector_observation[1]*vector_prediction[1]
        denominator0                = torch.sqrt(vector_observation[0]*vector_observation[0] + vector_observation[1]*vector_observation[1])
        denominator1                = torch.sqrt(vector_prediction[0]*vector_prediction[0] + vector_prediction[1]*vector_prediction[1])
        denominator                 = denominator0 * denominator1
        cosine                      = nominator / (denominator+0.01)
        cosine[torch.isnan(cosine)] = -0.01
        angle                       = torch.acos(cosine)#*180/math.pi 
        
    except:
        print("SINGULAR VALUE")
        sum_sigma_distance = torch.zeros(1) + math.pi/4
        return sum_sigma_distance

    return torch.mean(angle)


# ------------------------------------------------------------------------------
# Masking by segmentation
def mask_pedestrian_segmentation(segmentation):

    # tensor to img
    seg_frame       = (segmentation.permute(0,1,3,4,2).view(-1,image_size,image_size,image_dimension))#.numpy()) #PERMUTE to correct RGB space
    #seg_frame       = img_as_ubyte(seg_frame)

    # pool img
    avg_pool        = nn.AvgPool2d((int(image_size/mask_size), int(image_size/mask_size)), stride=(int(image_size/mask_size), int(image_size/mask_size)))
    #maskp           = torch.tensor(seg_frame[:,:,:,0].to(device), dtype=torch.double).view(-1,image_size,image_size)
    if device.type=='cuda':
            avg_pool.cuda()
    pooled_mask     = avg_pool(seg_frame[:,:,:,0].to(device)).view(-1, segmentation.size(1), mask_size, mask_size)

    return pooled_mask

# ------------------------------------------------------------------------------
# Encoder Model
class EncoderRNN(nn.Module):
    def __init__(self, in_size, embed_size, hidden_size, dropout_val=dropout_val, batch_size=1):
        super(EncoderRNN, self).__init__()
        
        # Configurations
        self.in_size                = in_size
        self.hidden_size            = hidden_size
        self.batch_size             = batch_size
        self.embed_size             = embed_size
        self.seq_length             = T_obs
        self.dropout_val            = dropout_val
        self.num_RRN_layers         = 1
        #self.vgg16_features_size    = vgg16_features_size
        self.visual_embed_size      = visual_embed_size

        #Architecture
        self.embedder_phi               = nn.Linear(self.in_size, self.embed_size)

        self.encoder                    = nn.LSTM(self.embed_size , self.hidden_size, self.num_RRN_layers, batch_first=True) # ezafe kon visual embedding_size ro
        self.dropout                    = nn.Dropout(dropout_val)
        self.embedder_out               = nn.Sequential(
                                                        nn.Linear(T_obs*hidden_size, hidden_size),
                                                        nn.ReLU(),
                                                        nn.Dropout(p=dropout_val),
                                                        nn.Linear(hidden_size, hidden_size),
                                                        nn.ReLU()
                                                        )
    
    def forward(self, input, hidden): 
        # Coordination Embedding
        embedding                   = self.embedder_phi(input.view(-1,2))
        embedding                   = F.relu(self.dropout(embedding))
        # RNN
        output, hidden              = self.encoder(embedding.unsqueeze(1), ( hidden[0],hidden[1] ) )
        return output, hidden


    def initHidden(self,batch_size):
        self.batch_size=batch_size
        return torch.zeros([self.num_RRN_layers, self.batch_size, self.hidden_size]).cuda()#, device = device)

    def emb_out(self,input):
        out= self.embedder_out(input)
        return out

# ------------------------------------------------------------------------------
# Decoder Model
class DecoderRNN(nn.Module):
    def __init__(self, in_size, embed_size, hidden_size, dropout_val=dropout_val, batch_size=1):
        super(DecoderRNN, self).__init__()
        
        # Configurations
        self.in_size                = in_size
        self.stochastic_out_size    = stochastic_out_size
        self.hidden_size            = hidden_size
        self.batch_size             = batch_size
        self.embed_size             = embed_size
        self.seq_length             = T_pred
        self.dropout_val            = dropout_val
        self.num_RRN_layers         = 1
        #self.vgg16_features_size    = vgg16_features_size
        self.visual_embed_size      = visual_embed_size
        #self.vgg16_features_size    = vgg16_features_size
        self.visual_embed_size      = visual_embed_size
        self.visual_size            = image_dimension * image_size * image_size

        #Architecture
        self.embedder_rho               = nn.Linear(self.in_size, self.embed_size)
        if startpoint_mode=="on":
            self.decoder                    = nn.LSTM(self.embed_size , self.hidden_size + self.hidden_size + in_size, self.num_RRN_layers, batch_first=True)
            self.fC_mu                      = nn.Sequential(
                                                            nn.Linear(self.hidden_size + self.hidden_size + in_size, int(self.hidden_size/2), bias=True),
                                                            nn.ReLU(),
                                                            nn.Dropout(p=dropout_val),
                                                            nn.Linear(int(self.hidden_size/2), self.stochastic_out_size, bias=True)
                                                            )
        else:
            self.decoder                    = nn.LSTM(self.embed_size , self.hidden_size + self.hidden_size , self.num_RRN_layers, batch_first=True)
            self.fC_mu                      = nn.Sequential(
                                                            nn.Linear(self.hidden_size + self.hidden_size , int(self.hidden_size/2), bias=True),
                                                            nn.ReLU(),
                                                            nn.Dropout(p=dropout_val),
                                                            nn.Linear(int(self.hidden_size/2), self.stochastic_out_size, bias=True)
                                                            )
        self.dropout                        = nn.Dropout(dropout_val)

        self.reducted_size = int((self.hidden_size-1)/3)+1
        if startpoint_mode =="on":
            self.reducted_size2 = int((self.hidden_size+in_size-1)/3)+1
            self.FC_dim_red                     = nn.Sequential(
                                                            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                                                            nn.Flatten(start_dim=1, end_dim=-1),
                                                            nn.Linear(self.reducted_size*self.reducted_size2, 2*self.hidden_size+in_size, bias=True),
                                                            nn.ReLU()
                                                            )
        else:
            self.FC_dim_red                     = nn.Sequential(
                                                            nn.MaxPool2d(kernel_size=3, stride=3, padding=1),
                                                            nn.Flatten(start_dim=1, end_dim=-1),
                                                            nn.Linear(self.reducted_size*self.reducted_size, 2*self.hidden_size, bias=True),
                                                            nn.ReLU()
                                                            )

    def forward(self, input, hidden): 
        # Coordination Embedding
        embedding                       = self.embedder_rho(input.view(-1,2))
        embedding                       = F.relu(self.dropout(embedding))
        output, hidden                  = self.decoder(embedding.unsqueeze(1), ( hidden[0],hidden[1] ))
        prediction                      = self.fC_mu(output.squeeze(0)) 
        return prediction, hidden #prediction_v.view(-1, self.visual_embed_size).cpu(), visual_embedding_ground_truth.cpu(), hidden


    def dim_red(self, input):
        output = self.FC_dim_red(input)
        return output


# ------------------------------------------------------------------------------
# A2Net Module
class A2Net(nn.Module):
    def __init__(self, in_channel, m, n, t, height, width, kernel_size, stride_size, batch_size=1):
        super(A2Net, self).__init__()
        self.m = m
        self.n = n
        self.width = width
        self.height = height
        self.t = t
        self.in_channel = in_channel
        self.Conv_Phi   = nn.Conv3d(in_channels=in_channel, out_channels=m, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Theta = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Rho   = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)
    
    def forward(self, input):
        input   = input.view(-1, self.in_channel, self.t, self.height, self.width)
        A       = self.Conv_Phi(input)
        B0      = self.Conv_Theta(input)
        # Softmax over thw dimension
        B       = F.softmax(B0.view(B0.size(0),B0.size(1),-1), dim=-1).view(B0.size()) 
        # 1st ATTN: Global Descriptors
        AB_T    = torch.einsum('bmthw, bnthw->bmn', A, B) 
        # 2nd ATTN: Attention Vectors
        # Softmax over n
        V       = F.softmax(self.Conv_Rho(input), dim=1)
        Z       = torch.einsum('bmn, bnthw->bmthw', AB_T, V)
        attn    = torch.einsum('bnthw, bnthw->bnhwt', B, V)
        return Z , attn#+input



# ------------------------------------------------------------------------------
# Conditional A2Net Module
class A2Net_Cond(nn.Module):
    def __init__(self, in_channel, m, n, t, height, width, kernel_size, stride_size, condition_size, batch_size=1):
        super(A2Net_Cond, self).__init__()
        self.m = m
        self.n = n
        self.width = width
        self.height = height
        self.t = t
        self.in_channel = in_channel
        self.Conv_Phi   = nn.Conv3d(in_channels=in_channel, out_channels=m, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Theta = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)
        self.Conv_Rho   = nn.Conv3d(in_channels=in_channel, out_channels=n, kernel_size=kernel_size, stride=stride_size)

        self.FC_Cond    = nn.Sequential(
                                nn.Linear(condition_size, condition_size, bias=True),
                                nn.ReLU(),
                                nn.Linear( condition_size, height*width, bias=True),
                                nn.ReLU()
                                )
    
    def forward(self, input, condition):
        input       = input.view(-1, self.in_channel, self.t, self.height, self.width)
        condition   = condition.view(input.size(0), -1)

        A       = self.Conv_Phi(input)
        B0      = self.Conv_Theta(input)
        # Softmax over thw dimension
        B       = F.softmax(B0.view(B0.size(0),B0.size(1),-1), dim=-1).view(B0.size())
        # Produce vector C from the condition
        self.start_point= condition[:,(-2,-1)] # trajectory encoded + start point
        C               = F.softmax(self.FC_Cond(condition), dim=-1).view(-1, self.height, self.width)
        self.condition  = C
        # 1st ATTN: Global Descriptors
        BC      = torch.einsum('bnthw, bhw->bnthw', B, C) 
        AB_T    = torch.einsum('bmthw, bnthw->bmn', A, BC) 
        # AB_TC   = torch.einsum('bmn, bm->bmn', AB_T, C)
        # 2nd ATTN: Attention Vectors
        # Softmax over n
        V       = F.softmax(self.Conv_Rho(input), dim=1)
        Z       = torch.einsum('bmn, bnthw->bmthw', AB_T, V)
        attn    = torch.einsum('bnthw, bnthw->bnhwt', B, V)
        return Z , attn#+input



# ------------------------------------------------------------------------------
# Vision Module
class Vision(nn.Module):
    def __init__(self, dropout_val=dropout_val, batch_size=1):
        super(Vision, self).__init__()
        k0 = s0 = 3
        p0 = 1
        self.CNN_0  = nn.Sequential(
                        nn.Conv3d(3, 16, kernel_size=k0, stride=[s0,s0,s0], padding=p0),
                        nn.ReLU(),
                        )
        # -----------
        m0 = 16
        n0 = 8#16
        k_a0 = s_a0 = 1
        h0 = w0 = int( (image_size - k0 + 2*p0)/s0 ) +1
        t0 = int( (T_obs - k0 + 2*p0)/s0 ) +1
        self.ATTN_0 = A2Net(16, m0, n0, t0, h0, w0, k_a0, s_a0)
        # -----------
        k1 = s1 = 3
        p1 = 1
        self.CNN_1  = nn.Sequential(
                        nn.Conv3d(16,16, kernel_size=k1, stride=[1,s1,s1], padding=p1),
                        nn.ReLU(),
                        nn.Conv3d(16,16, kernel_size=k1, stride=[1,s1,s1], padding=p1),
                        nn.ReLU()
                        )
        # -----------
        m1 = 16
        n1 = 8#32
        k_a1 = s_a1 = 1
        h1 = w1 = int( (h0 - k1 + 2*p1)/s1 ) +1
        h1 = w1 = int( (h1 - k1 + 2*p1)/s1 ) +1
        t1 = int( (t0 - k1 + 2*p1)/1 ) +1
        t1 = int( (t1 - k1 + 2*p1)/1 ) +1
        if startpoint_mode=="on":
            self.ATTN_1 = A2Net_Cond(16, m1, n1, t1, h1, w1, k_a1, s_a1, hidden_size+in_size)
        else:
            self.ATTN_1 = A2Net_Cond(16, m1, n1, t1, h1, w1, k_a1, s_a1, hidden_size)  
        # -----------
        k2 = 3
        s2 = 3
        p2 = 1
        self.CNN_2  = nn.Sequential(
                        nn.Conv3d(16, 16, kernel_size=k2, stride=[1,s2,s2], padding=p2),
                        nn.ReLU(),
                        )
        # -----------
        m2 = 16
        n2 = 8#8
        k_a2 = s_a2 = 1
        h2 = w2 = int( (h1 - k2 + 2*p2)/s2 ) +1
        t2 = int( (t1 - k2 + 2*p2)/1 ) +1
        if startpoint_mode=="on":
            self.ATTN_2 = A2Net_Cond(16, m2, n2, t2, h2, w2, k_a2, s_a2, hidden_size+in_size)  
        else:
            self.ATTN_2 = A2Net_Cond(16, m2, n2, t2, h2, w2, k_a2, s_a2, hidden_size)  
        # -----------
        # Global Average Pooling
        #self.GAP    = nn.AvgPool3d(kernel_size=[t2,h2,w2],stride=[t2,h2,w2])
        # -----------
        self.linear                             = nn.Sequential(
                                                    nn.Linear(16*h2*w2*t2 , hidden_size, bias=True),#+ hidden_size
                                                    nn.ReLU(),
                                                    nn.Dropout(p=dropout_val),
                                                    nn.Linear(hidden_size, hidden_size, bias=True),
                                                    nn.ReLU()
                                                    )    
        self.vsn_out_size = 16*h2*w2*t2#*16
        self.lastCNN_size = [h2, w2, t2]
        print("LAST conv size hwt\t", self.lastCNN_size)
        self.upsampling   = torch.nn.Upsample(size=(image_size,image_size,T_obs), mode='trilinear', align_corners=True)                 

    def forward(self, visual_input, condition):
         
        if startpoint_mode=="on":
            condition   = condition.view(-1, hidden_size + in_size)
        else:
            condition   = condition.view(-1, hidden_size)

        cc               = int(self.vsn_out_size/self.lastCNN_size[2]/self.lastCNN_size[0]/self.lastCNN_size[1])
        visual_input     = visual_input.view(-1,image_dimension,T_obs,image_size,image_size)
        cnn1             = self.CNN_0(visual_input)
        cnn2, attn2      = self.ATTN_0(cnn1)
        cnn3             = self.CNN_1(cnn2)
        cnn4, attn4      = self.ATTN_1(cnn3, condition)
        cnn5             = self.CNN_2(cnn4)
        cnn6, attn6      = self.ATTN_2(cnn5, condition)


        self.imgs        = cnn6.view(-1, cc, self.lastCNN_size[2], self.lastCNN_size[0], self.lastCNN_size[1]).requires_grad_(True)
        #cnn             = self.GAP(imgs)
        result           = self.linear(self.imgs.view(-1,self.vsn_out_size))
        # Calculate attn representations
        # weights         = self.linear[0].weight.sum(dim=0).view(self.vsn_out_size) #, dim=0)#weight[0]
        # attn_rep        = torch.einsum('bchwt,c->bhwt',imgs.detach(), weights.detach()).view(-1,1,self.lastCNN_size[0], self.lastCNN_size[1], self.lastCNN_size[2])
        # attn_rep_upsamp = self.upsampling(attn_rep).squeeze()


        return result, self.imgs, attn2,attn4,attn6#attn_rep_upsamp
    
    

# ------------------------------------------------------------------------------
# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, in_size, embed_size, hidden_size, dropout_val=dropout_val, batch_size=1):
        super(Seq2Seq, self).__init__()

        torch.cuda.empty_cache()
        self.encoder        = EncoderRNN(in_size, embed_size, hidden_size, dropout_val, batch_size=batch_size)
        self.encoder.apply(init_weights)
        
        self.decoder        = DecoderRNN(in_size, embed_size, hidden_size, dropout_val, batch_size=batch_size)
        self.decoder.apply(init_weights)
        
        self.vsn_module    = Vision(dropout_val, batch_size=batch_size)
        self.vsn_module.apply(init_weights)
        
        if device.type=='cuda':
            self.encoder.cuda()
            self.decoder.cuda()
            self.vsn_module.cuda()

    def forward(self,input_tensor, visual_input_tensor, output_tensor, batch_size, train_mode):
        batch_size      = int(input_tensor.size(0))#/torch.cuda.device_count())
        encoder_hidden  = (self.encoder.initHidden(batch_size),self.encoder.initHidden(batch_size))
        encoder_outputs = torch.zeros(batch_size, T_obs, hidden_size).cuda()#.cpu()
        start_point     = (input_tensor[:,0,:]).to(device).clone().detach()
        
        if startpoint_mode=="on":
            input_tensor[:,0,:]    = 0
        
        for t in range(0,T_obs):
            encoder_output, encoder_hidden  = self.encoder(input_tensor[:,t,:], encoder_hidden)
            encoder_outputs[:,t,:]          = encoder_output.squeeze(1)
        
        # Visual extraction/attention       
        # Enc outputs
        if enc_out=="on" and startpoint_mode=="on":
            encoder_extract             = self.encoder.emb_out(encoder_outputs.view(batch_size,-1))
            condition                   = torch.cat([encoder_extract.view(batch_size,-1),start_point.view(batch_size,-1)],dim=-1)
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,condition)
            decoder_hidden              = [torch.cat([encoder_extract.view(batch_size,-1),   visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        elif enc_out=="on" and startpoint_mode=="off":  
            encoder_extract             = self.encoder.emb_out(encoder_outputs.view(batch_size,-1))
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,encoder_extract)
            decoder_hidden              = [torch.cat([encoder_extract.view(batch_size,-1),   visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        elif enc_out=="off" and startpoint_mode=="on":
            condition = torch.cat([encoder_hidden[0].view(batch_size,-1),start_point.view(batch_size,-1)],dim=-1)
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,condition)
            decoder_hidden              = [torch.cat([encoder_hidden[0].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        else:
            visual_initial_vsn, attn_rep, attn2, attn4, attn6   = self.vsn_module(visual_input_tensor,encoder_hidden[0])
            decoder_hidden              = [torch.cat([encoder_hidden[0].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0),\
                                           torch.cat([encoder_hidden[1].view(batch_size,-1), visual_initial_vsn.view(batch_size,-1)],dim=-1).unsqueeze(0)]
        
        visual_vsn_result   = visual_initial_vsn

        # Initial Decoder input is current state coordinates
        # Initial Decoder hidden state is last Encoder hidden state

        decoder_input                   = input_tensor[:,-1,:]


        a0 = encoder_hidden[0].view(batch_size,-1)
        a1 = visual_vsn_result.view(batch_size,-1)
        a2 = torch.einsum("bn,bm->bnm",a0,a1)
        #tens_a = torch.cat([a0,a1,a2],dim=-1)
        tens_a = torch.ones(batch_size, a0.size(1)+1, a1.size(1)+1, device="cuda")
        tens_a[:,1:,1:] = a2
        tens_a[:,0,1:]  = a1
        tens_a[:,1:,0]  = a0


        b0 = encoder_hidden[1].view(batch_size,-1)
        b1 = visual_vsn_result.view(batch_size,-1)
        b2 = torch.einsum("bn,bm->bnm",b0,b1)
        # tens_b = torch.cat([b0,b1,b2],dim=-1)
        tens_b = torch.ones(batch_size, b0.size(1)+1, b1.size(1)+1, device="cuda")
        tens_b[:,1:,1:] = b2
        tens_b[:,0,1:]  = b1
        tens_b[:,1:,0]  = b0

        tens_a_red = self.decoder.dim_red(tens_a)
        tens_b_red = self.decoder.dim_red(tens_b)
        
        decoder_hidden                  = [tens_a_red.unsqueeze(0),\
                                           tens_b_red.unsqueeze(0)]

        
        # Tensor to store decoder outputs
        outputs                         = torch.zeros(batch_size, T_pred , in_size).cuda()#.cpu() 
        stochastic_outputs              = torch.zeros(batch_size, T_pred , stochastic_out_size).cuda()#.cpu()
        teacher_force                   = 1
        print('cuda:'+str(torch.cuda.current_device()))
        epsilonX                        = Normal(torch.zeros(batch_size,1),torch.ones(batch_size,1))
        epsilonY                        = Normal(torch.zeros(batch_size,1),torch.ones(batch_size,1))
        teacher_force                   = int(random.random() < teacher_forcing_ratio) if train_mode else 0
        print("Teacher Force\t",teacher_force)
        print("path mode\t",path_mode)
        for t in range(0, T_pred):
            stochastic_decoder_output, decoder_hidden   = self.decoder(decoder_input, decoder_hidden)
            # Reparameterization Trick :)
            decoder_output              = torch.zeros(batch_size,1,2).cuda()

            if stochastic_mode and path_mode=='single':
                decoder_output[:,:,0]  = stochastic_decoder_output[:,:,0] + epsilonX.sample().cuda() * stochastic_decoder_output[:,:,1]
                decoder_output[:,:,1]  = stochastic_decoder_output[:,:,2] + epsilonY.sample().cuda() * stochastic_decoder_output[:,:,3]
            elif stochastic_mode and path_mode=='avg':
                decoder_output[:,:,0]  = stochastic_decoder_output[:,:,0] + epsilonX.sample((avg_n_path_eval,1)).view(-1,avg_n_path_eval,1).mean(-2).cuda() * stochastic_decoder_output[:,:,1]
                decoder_output[:,:,1]  = stochastic_decoder_output[:,:,2] + epsilonY.sample((avg_n_path_eval,1)).view(-1,avg_n_path_eval,1).mean(-2).cuda() * stochastic_decoder_output[:,:,3]
            elif not(stochastic_mode):
                decoder_output[:,:,0]  = stochastic_decoder_output[:,:,0] 
                decoder_output[:,:,1]  = stochastic_decoder_output[:,:,2] 
            elif stochastic_mode and path_mode == "bst":
                epsilon_x               = torch.randn([batch_size,bst_n_path_eval,1], dtype=torch.float).cuda()
                epsilon_y               = torch.randn([batch_size,bst_n_path_eval,1], dtype=torch.float).cuda()
                multi_path_x            = stochastic_decoder_output[:,:,0].unsqueeze(1) + epsilon_x * stochastic_decoder_output[:,:,1].unsqueeze(1)
                multi_path_y            = stochastic_decoder_output[:,:,2].unsqueeze(1) + epsilon_y * stochastic_decoder_output[:,:,3].unsqueeze(1)
                ground_truth_x          = output_tensor[:,t,0].view(batch_size,1,1).cuda()
                ground_truth_y          = output_tensor[:,t,1].view(batch_size,1,1).cuda()
                diff_path_x             = multi_path_x - ground_truth_x
                diff_path_y             = multi_path_y - ground_truth_y
                diff_path               = (torch.sqrt( diff_path_x.pow(2) + diff_path_y.pow(2) )).sum(dim=-1)
                idx                     = torch.arange(batch_size,dtype=torch.long).cuda()
                min                     = torch.argmin(diff_path,dim=1).squeeze()
                decoder_output[:,:,0]   = multi_path_x[idx,min,:].view(batch_size,1)
                decoder_output[:,:,1]   = multi_path_y[idx,min,:].view(batch_size,1)
            elif stochastic_mode and path_mode == "top5":
                k = 5 #topk
                epsilon_x               = torch.randn([batch_size,bst_n_path_eval,1], dtype=torch.float).cuda()
                epsilon_y               = torch.randn([batch_size,bst_n_path_eval,1], dtype=torch.float).cuda()
                multi_path_x            = stochastic_decoder_output[:,:,0].unsqueeze(1) + epsilon_x * stochastic_decoder_output[:,:,1].unsqueeze(1)
                multi_path_y            = stochastic_decoder_output[:,:,2].unsqueeze(1) + epsilon_y * stochastic_decoder_output[:,:,3].unsqueeze(1)
                ground_truth_x          = output_tensor[:,t,0].view(batch_size,1,1).cuda()
                ground_truth_y          = output_tensor[:,t,1].view(batch_size,1,1).cuda()
                diff_path_x             = multi_path_x - ground_truth_x
                diff_path_y             = multi_path_y - ground_truth_y
                diff_path               = (torch.sqrt( diff_path_x.pow(2) + diff_path_y.pow(2) )).sum(dim=-1)
                idx                     = torch.arange(batch_size,dtype=torch.long).repeat(k).view(k,-1).transpose(0,1).cuda()
                min_val, min            = torch.topk(diff_path, k=k, dim=1,largest=False)
                decoder_output[:,:,0]   = multi_path_x[idx,min,:].mean(dim=-2).view(batch_size,1)
                decoder_output[:,:,1]   = multi_path_y[idx,min,:].mean(dim=-2).view(batch_size,1)

            # Log output
            outputs[:,t,:]                        = decoder_output.squeeze(1)
            stochastic_outputs[:,t,:]             = stochastic_decoder_output.squeeze(1)
            decoder_input                         = output_tensor[:,t,:] if teacher_force else decoder_output

        return outputs, stochastic_outputs, encoder_hidden, decoder_hidden, visual_vsn_result, attn_rep,attn2,attn4,attn6
            
            
# ------------------------------------------------------------------------------
# Train
def train(model, optimizer, scheduler, criterion, criterion_vision, clip,train_loader, validation_loader):
    global batch_size
    i               = None
    checked_frame   = 0

    print("Data Size ",data_size,"\tChunk Size ",chunk_size)
    global teacher_forcing_ratio
    counter =0
    for j in range(epoch_num):
        model.train()
        epoch_loss=0
        if j%7 == 6:
            teacher_forcing_ratio = (teacher_forcing_ratio - 0.2) if teacher_forcing_ratio>=0.1 else 0.0

        # Update TeachForce ratio to gradually change during training
        # global teacher_forcing_ratio
        # teacher_forcing_ratio-= 1/epoch_num
        print("TEACHER FORCE RATIO\t",teacher_forcing_ratio)
        #print("Learning Rate\t", scheduler.get_last_lr())

        if(j>=from_epoch):
            optimizer.zero_grad()
            start_time = time.time()
            ADE = 0
            FDE = 0
            i   = 0
            for i,data in enumerate( train_loader):
                print("\n--------------- Batch %d/ %d ---------------"%(j,i))  
                # Forward
                obs, pred, visual_obs, frame_tensor              = data
                input_tensor, output_tensor                      = obs.float().squeeze().to('cuda', non_blocking=True), pred.float().squeeze().to('cuda', non_blocking=True)               #(obs.to(device), pred.to(device))
                visual_input_tensor                              = visual_obs.squeeze().to('cuda', non_blocking=True)   #(visual_obs.to(device), visual_pred.to(device))
                prediction, stochastic_prediction, encoder_hidden, decoder_hidden, visual_embedding, attn_rep,_,_,_ = model(input_tensor,  visual_input_tensor, output_tensor, batch_size,train_mode=1)
                

                calculated_prediction = prediction.cumsum(axis=1) 

                loss_line_regularizer                            = distance_from_line_regularizer(input_tensor,calculated_prediction)
                
                if biased_loss_mode:
                    weight  = torch.arange(1,2*T_pred+1,2).cuda().float()
                    weight  = torch.exp(weight / T_pred).repeat(prediction.size(0)).view(prediction.size(0),T_pred,1)
                    loss    = criterion( (calculated_prediction)*weight, torch.cumsum(output_tensor,dim=-2)*weight)
                else:
                    loss    = criterion( (calculated_prediction), torch.cumsum(output_tensor,dim=-2))       
                out_x           = output_tensor[:,:,0].cumsum(axis=1)
                out_y           = output_tensor[:,:,1].cumsum(axis=1)
                pred_x          = calculated_prediction[:,:,0]
                pred_y          = calculated_prediction[:,:,1]
                ADE             += ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0).mean(0)   
                # FDE             += ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0)[-1]
                # Backward Propagation

                total_loss      = loss.double() + torch.tensor(regularization_factor).to('cuda', non_blocking=True) * loss_line_regularizer.double()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                print("Total Loss\t{:.2f}".format(total_loss.item()))
                epoch_loss += total_loss.item()
                print("Time\t\t{:.2f} sec \n".format(time.time() - start_time))
                start_time = time.time()
                torch.cuda.empty_cache()
                writer.close()
                count_div=i
            
            # tensorboard log
            writer.add_scalar('ADE/train', ADE.item()/(count_div+1),     counter )
            # writer.add_scalar('FDE/train', FDE.item()/(count_div+1),     counter )
            # writer.add_scalar('LOSS/train', epoch_loss/(count_div+1)   , counter)
            counter += 1

        if scheduler.get_last_lr()[0]>0.001:
            scheduler.step()
        # validation(model, optimizer, criterion, criterion_vision, clip, validation_loader, j) 
        print("EPOCH ", j, "\tLOSS ",epoch_loss / (int(data_size/chunk_size)))
        writer.add_scalar('epoch_loss/train', epoch_loss/ (int(data_size/chunk_size)), j )
        torch.save( model.state_dict(), model_path+"_current")
        print("-----------------------------------------------\n"+"-----------------------------------------------")
    return epoch_loss / (int(data_size/chunk_size))


# ------------------------------------------------------------------------------
# Evaluate
def validation(model, optimizer, criterion, criterion_vision, clip, validation_loader, counter):
    global batch_size
    model.eval()
    i           = None
    ADEs        = 0
    FDEs        = 0
    epoch_loss  = 0
    loss_line_regularizer = 0
    loss = 0 
    total_loss = 0
    ADE  = 0
    FDE  = 0
    for i,data in enumerate( test_loader):
        # Forward
        obs, pred, visual_obs, frame_tensor                 = data
        input_tensor, output_tensor                         = obs.float().squeeze().to('cuda', non_blocking=True), pred.float().squeeze().to('cuda', non_blocking=True)               #(obs.to(device), pred.to(device))
        visual_input_tensor                                 = visual_obs.squeeze().to('cuda', non_blocking=True)   #(visual_obs.to(device), visual_pred.to(device))
        prediction, stochastic_prediction, encoder_hidden, decoder_hidden, visual_embedding, attn_rep,_,_,_ = model(input_tensor, visual_input_tensor, output_tensor, batch_size, train_mode=0)
        
        calculated_prediction = prediction.cumsum(axis=1) 

        loss_line_regularizer           = distance_from_line_regularizer(input_tensor,calculated_prediction)
        
        if biased_loss_mode:
            weight  = torch.arange(1,2*T_pred+1,2).cuda().float()
            weight  = torch.exp(weight / T_pred).repeat(prediction.size(0)).view(prediction.size(0),T_pred,1)
            loss    = criterion( (calculated_prediction)*weight, torch.cumsum(output_tensor,dim=-2)*weight)
        else:
            loss    = criterion( (calculated_prediction), torch.cumsum(output_tensor,dim=-2))
        out_x           = output_tensor[:,:,0].cumsum(axis=1)
        out_y           = output_tensor[:,:,1].cumsum(axis=1)
        pred_x          = calculated_prediction[:,:,0]
        pred_y          = calculated_prediction[:,:,1]
        ADE             += ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0).mean(0)   
        FDE             += ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0)[-1]
        total_loss      += loss.double() + regularization_factor * loss_line_regularizer.double() 
        print("Total Loss\t{:.2f}".format(total_loss.item()))

    writer.add_scalar('ADE/val_'+path_mode,             ADE.item()/(i+1),             counter )
    writer.add_scalar('FDE/val_'+path_mode,             FDE.item()/(i+1),             counter )
    writer.add_scalar('LOSS/val_'+path_mode,            total_loss.item()/(i+1)   ,   counter )
    writer.add_scalar('LOSS_c/val_'+path_mode,          loss.item()/(i+1)        ,    counter )
    writer.add_scalar('L-REGULARIZER/val_'+path_mode,   loss_line_regularizer.item()/(i+1), counter )
    writer.close()


# ------------------------------------------------------------------------------
# Evaluate
def evaluate_eval(model, optimizer, criterion, criterion_vision, clip, five_fold_cross_validation):
    global batch_size
    model.eval()
    i           = None
    ADEs        = 0
    FDEs        = 0
    epoch_loss  = 0
    list_x_obs          = ['x_obs_'+str(i)              for i in range(0,T_obs)]
    list_y_obs          = ['y_obs_'+str(i)              for i in range(0,T_obs)]
    list_c_context      = ['context_c_'+str(i)          for i in range(0,hidden_size)]
    list_h_context      = ['context_h_'+str(i)          for i in range(0,hidden_size)]
    list_x_pred         = ['x_pred_'+str(i)             for i in range(0,T_pred)]
    list_y_pred         = ['y_pred_'+str(i)             for i in range(0,T_pred)]
    list_x_stoch_pred_m = ['x_stoch_pred_m_'+str(i)     for i in range(0,T_pred)]
    list_y_stoch_pred_m = ['y_stoch_pred_m_'+str(i)     for i in range(0,T_pred)]
    list_x_stoch_pred_s = ['x_stoch_pred_s_'+str(i)     for i in range(0,T_pred)]
    list_y_stoch_pred_s = ['y_stoch_pred_s_'+str(i)     for i in range(0,T_pred)]
    list_x_out          = ['x_out_'+str(i)              for i in range(0,T_pred)]
    list_y_out          = ['y_out_'+str(i)              for i in range(0,T_pred)]
    list_vsn           = ['vsn_'+str(i)               for i in range(0,hidden_size)]
    # list_vsn_visual    = ['vsn_vis_'+str(i)           for i in range(0,T_obs*T_pred)]
    df_out              = pd.DataFrame(columns=list_x_obs + list_y_obs + list_x_out + list_y_out + list_x_pred + list_y_pred + list_x_stoch_pred_m + list_y_stoch_pred_m + list_x_stoch_pred_s + list_y_stoch_pred_s + list_c_context + list_h_context + list_vsn)# + list_vsn_visual)

    for i,data in enumerate( test_loader):
        start_time = time.time()
        # Forward
        obs, pred, visual_obs, frame_tensor                 = data
        input_tensor, output_tensor                         = obs.float().squeeze().to('cuda', non_blocking=True), pred.float().squeeze().to('cuda', non_blocking=True)               #(obs.to(device), pred.to(device))
        visual_input_tensor                                 = visual_obs.squeeze().cuda()   #(visual_obs.to(device), visual_pred.to(device))
        prediction, stochastic_prediction, encoder_hidden, decoder_hidden, visual_embedding, attn_rep, attn2, attn4, attn6 = model(input_tensor, visual_input_tensor, output_tensor, batch_size, train_mode=0)
    

        calculated_prediction =  prediction.cumsum(axis=1) 

        loss_line_regularizer                               = distance_from_line_regularizer(input_tensor,calculated_prediction)

        if biased_loss_mode:
            weight  = torch.arange(1,2*T_pred+1,2).cuda().float()
            weight  = torch.exp(weight / T_pred).repeat(prediction.size(0)).view(prediction.size(0),T_pred,1)
            loss    = criterion( (calculated_prediction)*weight, torch.cumsum(output_tensor,dim=-2)*weight)
        else:
            loss    = criterion( (calculated_prediction), torch.cumsum(output_tensor,dim=-2))
        out_x           = output_tensor[:,:,0].cumsum(axis=1)
        out_y           = output_tensor[:,:,1].cumsum(axis=1)
        pred_x          = calculated_prediction[:,:,0]
        pred_y          = calculated_prediction[:,:,1]
        ADE             = ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0).mean(0)   
        FDE             = ((out_x.sub(pred_x)**2).add((out_y.sub(pred_y)**2))**(1/2)).mean(0)[-1]
        total_loss      = loss.double() + regularization_factor * loss_line_regularizer.double() 
        print("Total Loss\t{:.2f}".format(total_loss.item()))
        epoch_loss += total_loss.item()
        ADEs    += ADE.item()
        FDEs    += FDE.item()
        input_x_lin                 = input_tensor[:,:,0].view(-1, T_obs).cpu()
        input_y_lin                 = input_tensor[:,:,1].view(-1, T_obs).cpu()
        output_x_lin                = output_tensor[:,:,0].view(-1, T_pred).cpu()
        output_y_lin                = output_tensor[:,:,1].view(-1, T_pred).cpu()
        prediction_x_lin            = prediction[:,:,0].view(-1, T_pred).cpu()
        prediction_y_lin            = prediction[:,:,1].view(-1, T_pred).cpu()
        stoch_prediction_x_m        = stochastic_prediction[:,:,0].view(-1, T_pred).cpu()
        stoch_prediction_x_s        = stochastic_prediction[:,:,1].view(-1, T_pred).cpu()
        stoch_prediction_y_m        = stochastic_prediction[:,:,2].view(-1, T_pred).cpu()
        stoch_prediction_y_s        = stochastic_prediction[:,:,3].view(-1, T_pred).cpu()
        context_h_lin               = encoder_hidden[0].view(-1, hidden_size).cpu()
        context_c_lin               = encoder_hidden[1].view(-1, hidden_size).cpu()
        visual_embedding_weights    = visual_embedding.view(-1, hidden_size).cpu()

        whole_data                  = torch.cat((input_x_lin, input_y_lin, output_x_lin, output_y_lin, prediction_x_lin, prediction_y_lin, stoch_prediction_x_m, stoch_prediction_y_m, stoch_prediction_x_s, stoch_prediction_y_s, context_c_lin, context_h_lin, visual_embedding_weights), 1)
        temp                        = pd.DataFrame(whole_data.detach().cpu().numpy(), columns=list_x_obs + list_y_obs + list_x_out + list_y_out + list_x_pred + list_y_pred + list_x_stoch_pred_m + list_y_stoch_pred_m + list_x_stoch_pred_s + list_y_stoch_pred_s + list_c_context + list_h_context + list_vsn)
        df_out                      = df_out.append(temp)
        df_out.reset_index(drop=True,inplace=True)

        print("Time\t\t{:.2f} sec \n".format(time.time() - start_time))


    # ADE/FDE Report
    out_x  = df_out[['x_out_' +str(i) for i in range(0,T_pred)]].cumsum(axis=1)
    pred_x = df_out[['x_pred_'+str(i) for i in range(0,T_pred)]].cumsum(axis=1)
    out_y  = df_out[['y_out_' +str(i) for i in range(0,T_pred)]].cumsum(axis=1)
    pred_y = df_out[['y_pred_'+str(i) for i in range(0,T_pred)]].cumsum(axis=1)
    ADE = (out_x.sub(pred_x.values)**2).add((out_y.sub(pred_y.values)**2).values, axis=1)**(1/2)
    df_out['ADE'] = ADE.mean(axis=1)
    FDE = ADE.x_out_11
    df_out['FDE'] = FDE
    Mean_ADE = df_out.ADE.mean()
    Mean_FDE = df_out.FDE.mean()
    print("MEAN ADE/FDE\t",Mean_ADE,Mean_FDE)
    writer.add_scalar("Final_Test/ADE_"+path_mode, Mean_ADE, global_step=0)
    writer.add_scalar("Final_Test/FDE_"+path_mode, Mean_FDE, global_step=0)

    df_out.to_sql(table_out+'_'+path_mode, cnx2, if_exists="replace", index=False)
    writer.close()
    return ADEs, FDEs, int(data_size/chunk_size)


# ------------------------------------------------------------------------------
# Run
model                       = Seq2Seq(in_size, embed_size, hidden_size, dropout_val=dropout_val, batch_size=batch_size)
model                       = nn.DataParallel( model ).cuda()

learning_step               = 40
initial_learning_rate       = 0.01
clip                        = 1
# MSE loss
criterion                   = nn.MSELoss(reduction='mean')#nn.NLLLoss()
criterion_vision            = nn.MSELoss(reduction='sum')#nn.NLLLoss()
# SGD optimizer
optimizer                   = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.01)
scheduler                   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_step, gamma=0.1)
five_fold_cross_validation  = 0

dataset_train = TrajectoryPredictionDataset(image_folder_path, DB_PATH_train, cnx_train)

train_loader        = torch.utils.data.DataLoader(dataset_train,        batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)
validation_loader   = None

# ----------------------------------------------------------
# print("LOAD MODEL")
# initial_model_path="** path to model **/NNmodel/model_current"
# checkpoint = torch.load(initial_model_path,map_location=device)
# model.load_state_dict(checkpoint)

print("TRAIN")
model.train()
print("path mode\t",path_mode)
loss               = train(model, optimizer, scheduler, criterion, criterion_vision, clip, train_loader, validation_loader)
print("LOSS ",loss)

# ----------------------------------------------------------

print("SAVE MODEL")
torch.save( model.state_dict(), model_path)
print("LOAD MODEL")
# Change device to cpu
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()



model               = Seq2Seq(in_size, embed_size, hidden_size, dropout_val=dropout_val, batch_size=batch_size)
model               = nn.DataParallel( model ).cuda()
checkpoint          = torch.load(model_path,map_location=device)
model.load_state_dict(checkpoint)

# ----------------------------------------------------------
#TEST DATASET AND LOADER

dataset_val   = TrajectoryPredictionDataset(image_folder_path, DB_PATH_val, cnx_val)
test_loader   = torch.utils.data.DataLoader(dataset_val,         batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

print("EVALUATE")
model.eval()
path_mode = 'bst'
print("path mode\t",path_mode)
evaluate_eval(model, optimizer, criterion, criterion_vision, clip, test_loader)

print("EVALUATE")
model.eval()
path_mode = 'top5'
print("path mode\t",path_mode)
evaluate_eval(model, optimizer, criterion, criterion_vision, clip, test_loader)

