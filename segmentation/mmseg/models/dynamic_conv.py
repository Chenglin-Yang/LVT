import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class attention1d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention1d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv1d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv1d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention1d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height = x.size()
        x = x.view(1, -1, height, )# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-1))
        return output



class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output

    
class maker(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(maker, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=True)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class dw_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True, **kwargs):
        super().__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        self.bias = bias
        self.K = K
        self.cfg = kwargs['cfg']
        self.maker = maker(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            if self.cfg.dynamic_weight.init == 'dirac':
                nn.init.dirac_(self.weight[i], self.groups)
            elif self.cfg.dynamic_weight.init == 'kaiming':
                nn.init.kaiming_uniform_(self.weight[i])
            else:
                raise NotImplementedError
        if self.with_bias:
            if self.cfg.dynamic_weight.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.cfg.dynamic_weight.init == 'kaiming':
                bound = self.groups / (self.kernel_size**2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError
            
    def update_temperature(self):
        self.maker.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        
        softmax_attention = self.maker(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
#         aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        aggregate_weight = torch.mm(softmax_attention, weight).view(
            batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output
    
    
class ds_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, 
                 dilation=[1,3,5], groups=1, bias=True, init_weight=True, **kwargs):
        super().__init__()
        assert in_planes%groups==0
        assert kernel_size==3, 'need to rewrite the calculation of padding'
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.with_bias = bias
        self.cfg = kwargs['cfg']
        
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_planes))
        else:
            self.bias = None
        
        self.silu = nn.SiLU(True)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
            
        if self.cfg.dynamic_scale.init == 'dirac':
            nn.init.dirac_(self.weight, self.groups)
        elif self.cfg.dynamic_scale.init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight)
        else:
            raise NotImplementedError
        
        if self.with_bias:
            if self.cfg.dynamic_scale.init == 'dirac':
                nn.init.constant_(self.bias, 0.)
            elif self.cfg.dynamic_scale.init == 'kaiming':
                bound = self.groups / (self.kernel_size**2 * self.in_planes)
                bound = math.sqrt(bound)
                nn.init.uniform_(self.bias, -bound, bound)
            else:
                raise NotImplementedError

    def forward(self, x): #将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        
        output = 0
        for dil in self.dilation:
            output += self.silu(
                F.conv2d(
                    x, weight=self.weight, bias=self.bias, stride=self.stride, padding=dil,
                    dilation=dil, groups=self.groups,
                )
            )
        return output

class dsc(nn.Module):
    def __init__(
        self,
        cfg,
        in_planes, 
        out_planes, 
        kernel_size=3,  
        stride=1, 
        bias=True,
        init_weight=True, 
    ):
        super().__init__()
        self.cfg = cfg

        if self.cfg.first_block == 'dynamic_weight':
            dw_in_planes = in_planes
            dw_out_planes = in_planes
            ds_in_planes = in_planes
            ds_out_planes = out_planes
        elif self.cfg.first_block == 'dynamic_scale':
            dw_in_planes = in_planes
            dw_out_planes = out_planes
            ds_in_planes = in_planes
            ds_out_planes = in_planes
        else:
            raise NotImplementedError

        if self.cfg.dynamic_weight.base is None:
            self.dynamic_weight = nn.Conv2d(
                in_channels=dw_in_planes, 
                out_channels=dw_out_planes,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1, 
                groups=1, 
                bias=bias,
            )
        else:
            self.dynamic_weight = dw_conv2d(
                cfg=self.cfg,
                in_planes=dw_in_planes, 
                out_planes=dw_out_planes, 
                kernel_size=1, 
                ratio=1/self.cfg.dynamic_weight.cr,
                stride=1, 
                padding=0,
                dilation=1, 
                groups=1,
                bias=bias, 
                K=self.cfg.dynamic_weight.base,
                temperature=self.cfg.dynamic_weight.temperature, 
                init_weight=init_weight, 
            )
        
        if self.cfg.dynamic_scale.base is None:
            self.dynamic_scale = nn.Conv2d(
                in_channels=ds_in_planes, 
                out_channels=ds_out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                dilation=1, 
                groups=ds_in_planes, 
                bias=bias,
            )
        else:
            self.dynamic_scale = ds_conv2d(
                cfg=self.cfg,
                in_planes=ds_in_planes, 
                out_planes=ds_out_planes, 
                kernel_size=kernel_size, 
                stride=stride,
                dilation=self.cfg.dynamic_scale.base, 
                groups=ds_in_planes, 
                bias=bias, 
                init_weight=init_weight, 
            )
        self.apply(self._init_weights)
        
    def forward(self, x):
        if self.cfg.first_block == 'dynamic_weight':
            x = self.dynamic_scale(self.dynamic_weight(x))
        elif self.cfg.first_block == 'dynamic_scale':
            x = self.dynamic_weight(self.dynamic_scale(x))
        else:
            raise NotImplementedError
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)

class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None


        #TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output


if __name__ == '__main__':
    x = torch.randn(24, 3,  20)
    model = Dynamic_conv1d(in_planes=3, out_planes=16, kernel_size=3, ratio=0.25, padding=1,)
    x = x.to('cuda:0')
    model.to('cuda')
    # model.attention.cuda()
    # nn.Conv3d()
    print(model(x).shape)
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    model.update_temperature()
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)
    print(model(x).shape)


