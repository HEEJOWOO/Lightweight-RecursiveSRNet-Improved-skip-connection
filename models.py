from torch import nn
import torch
import torch.nn.functional as F
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
    
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class SB_layer(nn.Module): #Small Block Layer
    def __init__(self, growth_rate, i):
        super(SB_layer, self).__init__()
        gr = growth_rate
        self.layer_1x1_3x3 = nn.Sequential(nn.Conv2d(gr + i * gr, gr, 1),nn.ReLU(True), nn.Conv2d(gr, gr, 3, padding=1, bias=True))
        
    def forward(self, x):
        layer_1x1_3x3 = self.layer_1x1_3x3(x)
        return layer_1x1_3x3

class SB_layer_residual(nn.Module): #Small Block Layer
    def __init__(self, growth_rate):
        super(SB_layer_residual, self).__init__()
        gr = growth_rate
        self.layer_3x3 = nn.Conv2d(gr,gr,3,1,1)
        
    def forward(self, x):
        layer_3x3 = self.layer_3x3(x)
        return layer_3x3

class Refine_layer(nn.Module): #Information Refinement Layer
    def __init__(self, growth_rate,distillation_rate):
        super(Refine_layer, self).__init__()
        gr = growth_rate
        rf = growth_rate-(int(growth_rate*distillation_rate)) #48
        self.layer_3x3_3x3 = nn.Sequential(nn.Conv2d(rf,gr,3,1,1),nn.ReLU(True),nn.Conv2d(gr,rf,3,1,1))
        self.layer_3x3 = nn.Conv2d(rf,rf,3,1,1)
    def forward(self, x):
        layer_3x3_3x3 = self.layer_3x3_3x3(x)+self.layer_3x3(x)
        return layer_3x3_3x3  

class Retain_layer(nn.Module): #Information Retainment Layer
    def __init__(self, growth_rate,distillation_rate):
        super(Retain_layer, self).__init__()
        gr = growth_rate
        rt = int(growth_rate*distillation_rate)
        self.layer_3x3 = nn.Conv2d(rt+gr,gr//2,3,1,1)

    def forward(self, x):
        layer_3x3 = self.layer_3x3(x)
        return layer_3x3   
        
class RDCAB(nn.Module):
    def __init__(self, in_channels, growth_rate,distillation_rate):
        super(RDCAB, self).__init__()
        reduction_ratio = 16
        gr = growth_rate
        dr = distillation_rate

        self.distilled_channels = int(gr*dr) 
        self.remaining_channels = int(gr-(gr*dr))
        
        self.SB_layer_1 = SB_layer(gr, 0)
        self.SB_layer_residual_1 = SB_layer_residual(gr)
        self.Refine_layer_1 = Refine_layer(gr,dr)
        self.Retain_layer_1 = Retain_layer(gr,dr)
        
        self.SB_layer_2 = SB_layer(gr, 1)
        self.SB_layer_residual_2 = SB_layer_residual(gr)
        self.Refine_layer_2 = Refine_layer(gr,dr)
        self.Retain_layer_2 = Retain_layer(gr,dr)
        
        self.SB_layer_3 = SB_layer(gr, 2)
        self.SB_layer_residual_3 = SB_layer_residual(gr)
        self.Refine_layer_3 = Refine_layer(gr,dr)
        self.Retain_layer_3 = Retain_layer(gr,dr)
        
        self.SB_layer_4 = SB_layer(gr, 3)
        self.SB_layer_residual_4 = SB_layer_residual(gr)
        self.Refine_layer_4 = Refine_layer(gr,dr)
        self.Retain_layer_4 = Retain_layer(gr,dr)
        
        self.SB_layer_5 = SB_layer(gr, 4)
        self.SB_layer_residual_5 = SB_layer_residual(gr)
        self.Refine_layer_5 = Refine_layer(gr,dr)
        self.Retain_layer_5 = Retain_layer(gr,dr)
        
        self.SB_layer_6 = SB_layer(gr, 5)
        self.SB_layer_residual_6 = SB_layer_residual(gr)
        self.Refine_layer_6 = Refine_layer(gr,dr)
        self.Retain_layer_6 = Retain_layer(gr,dr)
        
        self.SB_layer_7 = SB_layer(gr, 6)
        self.SB_layer_residual_7 = SB_layer_residual(gr)
        self.Refine_layer_7 = Refine_layer(gr,dr)
        self.Retain_layer_7 = Retain_layer(gr,dr)

        self.SB_layer_8 = SB_layer(gr, 7)
        self.SB_layer_residual_8 = SB_layer_residual(gr)
        self.Refine_layer_8 = nn.Sequential(nn.Conv2d(self.remaining_channels,gr,3,1,1),nn.ReLU(True),nn.Conv2d(gr,gr//2,3,1,1))

        self.lff = nn.Conv2d((gr//2)*8, gr, kernel_size=1)
        
 
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(64, 64 // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 // 16, 64, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        

    def forward(self, x, sfe_residual): 
        Local_Residual = x
        
        SB_layer_1 = self.SB_layer_1(x)+self.SB_layer_residual_1(x)
        distilled_c1, remaining_c1 = torch.split(SB_layer_1, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_1 = self.Refine_layer_1(remaining_c1)
        ID_concat_1 = torch.cat([distilled_c1,Refine_layer_1],1)
        LFF_layer_1 = self.Retain_layer_1(torch.cat([SB_layer_1,distilled_c1],1))

        SB_layer_2 = self.SB_layer_2(torch.cat((x, ID_concat_1), 1))+self.SB_layer_residual_2(ID_concat_1)
        distilled_c1, remaining_c1 = torch.split(SB_layer_2, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_2 = self.Refine_layer_2(remaining_c1)
        ID_concat_2 = torch.cat([distilled_c1,Refine_layer_2],1)
        LFF_layer_2 = self.Retain_layer_2(torch.cat([SB_layer_2,distilled_c1],1))
        
        SB_layer_3 = self.SB_layer_3(torch.cat((x, ID_concat_1,ID_concat_2), 1))+self.SB_layer_residual_3(ID_concat_2)
        distilled_c1, remaining_c1 = torch.split(SB_layer_3, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_3 = self.Refine_layer_3(remaining_c1)
        ID_concat_3 = torch.cat([distilled_c1,Refine_layer_3],1)
        LFF_layer_3 = self.Retain_layer_3(torch.cat([SB_layer_3,distilled_c1],1))
        
        SB_layer_4 = self.SB_layer_4(torch.cat((x, ID_concat_1,ID_concat_2,ID_concat_3), 1))+self.SB_layer_residual_4(ID_concat_3)
        distilled_c1, remaining_c1 = torch.split(SB_layer_4, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_4 = self.Refine_layer_4(remaining_c1)
        ID_concat_4 = torch.cat([distilled_c1,Refine_layer_4],1)
        LFF_layer_4 = self.Retain_layer_4(torch.cat([SB_layer_4,distilled_c1],1))
        
        SB_layer_5 = self.SB_layer_5(torch.cat((x, ID_concat_1,ID_concat_2,ID_concat_3,ID_concat_4), 1))+self.SB_layer_residual_5(ID_concat_4)
        distilled_c1, remaining_c1 = torch.split(SB_layer_5, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_5 = self.Refine_layer_5(remaining_c1)
        ID_concat_5 = torch.cat([distilled_c1,Refine_layer_5],1)
        LFF_layer_5 = self.Retain_layer_5(torch.cat([SB_layer_5,distilled_c1],1))
        
        SB_layer_6 = self.SB_layer_6(torch.cat((x, ID_concat_1,ID_concat_2,ID_concat_3,ID_concat_4,ID_concat_5), 1))+self.SB_layer_residual_6(ID_concat_5)
        distilled_c1, remaining_c1 = torch.split(SB_layer_6, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_6 = self.Refine_layer_6(remaining_c1)
        ID_concat_6 = torch.cat([distilled_c1,Refine_layer_6],1)
        LFF_layer_6 = self.Retain_layer_6(torch.cat([SB_layer_6,distilled_c1],1))
        
        SB_layer_7 = self.SB_layer_7(torch.cat((x, ID_concat_1,ID_concat_2,ID_concat_3,ID_concat_4,ID_concat_5,ID_concat_6), 1))+self.SB_layer_residual_7(ID_concat_6)
        distilled_c1, remaining_c1 = torch.split(SB_layer_7, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_7 = self.Refine_layer_7(remaining_c1)
        ID_concat_7 = torch.cat([distilled_c1,Refine_layer_7],1)
        LFF_layer_7 = self.Retain_layer_7(torch.cat([SB_layer_7,distilled_c1],1))
        
        SB_layer_8 = self.SB_layer_8(torch.cat((x, ID_concat_1,ID_concat_2,ID_concat_3,ID_concat_4,ID_concat_5,ID_concat_6,ID_concat_7), 1))+self.SB_layer_residual_8(ID_concat_7)
        distilled_c1, remaining_c1 = torch.split(SB_layer_8, (self.distilled_channels, self.remaining_channels), dim=1) # 16 48
        Refine_layer_8 = self.Refine_layer_8(remaining_c1)
        
        out = torch.cat([LFF_layer_1,LFF_layer_2,LFF_layer_3,LFF_layer_4,LFF_layer_5,LFF_layer_6,LFF_layer_7,Refine_layer_8], dim=1) 
        
        x = self.lff(out)
        y =self.contrast(x)+self.avg_pool(x)
        y = self.conv_du(y)
        x = x*y
        x = x+Local_Residual
        
        return x


class recursive_SR(nn.Module):
    def __init__(self,num_channels, num_features, growth_rate,U, distillation_rate):
        super(recursive_SR, self).__init__()
        self.U = U
        self.G0 = num_features
        self.G = growth_rate
        self.dr = distillation_rate
        self.rdbs = RDCAB(self.G0, self.G,self.dr) 
        
    def forward(self, sfe2):
        global cocnat_LF
        x=sfe2
        local_features = []
        for i in range(self.U):
            x = self.rdbs(x)
            local_features.append(x)
        cocnat_LF = torch.cat(local_features, 1)
        return x
        
class Net(nn.Module):
    def __init__(self, scale_factor, num_channels, num_features, growth_rate, U,distillation_rate):
        super(Net, self).__init__()
        self.scale_factor=scale_factor
        self.G0 = num_features
        self.G = growth_rate
        self.U = U
        
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        self.rbs = nn.Sequential(*[recursive_SR(num_features,num_features,growth_rate,U,distillation_rate)])
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.U, self.G0, kernel_size=1),nn.ReLU(True),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        
        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )
        # middle point wise 64->32
        self.middle_pointwise=nn.Conv2d(self.G0,32,1)
        # information refinement block
        self.convIRB_1 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB_2 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB_3 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB_4 = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1,groups=1,bias=False),nn.ReLU(inplace=True))
        self.convIRB = nn.Sequential(nn.Conv2d(in_channels=32,out_channels=3,kernel_size=3,padding=1,groups=1,bias=False))

    def forward(self, x):
        x_up = F.interpolate(x, mode='bicubic',scale_factor=self.scale_factor)
        #shallow feature extraction
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        #recursive feature extraction
        x= self.recursive_SR(sfe2)
        x = self.gff(concat_LF) + sfe1
        x = self.upscale(x)
        x = self.middle_pointwise(x)
        #information refinement block
        x = self.convIRB_1(x)
        x = self.convIRB_2(x)
        x = self.convIRB_3(x)
        x = self.convIRB_4(x)
        x = self.convIRB(x)+x_up
        
        return x
        
