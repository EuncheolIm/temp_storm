import numpy as np
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def lmake_xy( dir_file):
        xy = np.genfromtxt(dir_file, delimiter=',', dtype = np.float64)
        # print(xy.shape)
        return xy
    
def lmake_xy1( dir_file):
        xy = np.genfromtxt(dir_file, delimiter='', dtype = np.float64)
        # print(xy.shape)
        return xy

## 이 데이터는 0.02 주기로 log했기 때문에 실제로는 data.shape[0] *20 해야함--> 약 10s
data = lmake_xy("/home/kist/TRO_plot_working/extended_comapre_PM.txt")

# data = lmake_xy("/home/kist/바탕화면/storm/examples/data_inter3.txt")
# data1 = lmake_xy("/home/kist/바탕화면/storm/examples/data2.txt")

# data1 = lmake_xy("/home/kist/바탕화면/storm/examples/compare_PM.txt")
# print("data: ",data1.shape)
# original_timestamps = np.linspace(0, 0.02 * 500, 501)
# new_timestamps = np.linspace(0, 0.02 * 500, 501 * 20)
# # 선형 보간 함수 생성
# linear_interp = interp1d(original_timestamps, data1, axis=0)

# # 새로운 타임스탬프에 대해 데이터 보간
# interpolated_data = linear_interp(new_timestamps)
# print(interpolated_data.shape[0])

# np.savetxt("data_inter3.txt", interpolated_data, delimiter=",")

# pm_data = lmake_xy1("/home/kist/바탕화면/storm/examples/PM_data.txt")

# pm_data = lmake_xy1("/home/kist/바탕화면/storm/examples/PM_data2.txt")
pm_data = lmake_xy1("/home/kist/바탕화면/storm/examples/PM_data516.txt") # time test
ddp_data = lmake_xy("/home/kist/바탕화면/storm/examples/data_ddp31.txt")
# ddp_data = lmake_xy("/home/kist/바탕화면/storm/examples/data_ddp_1kHz.txt")

# ddp_data = lmake_xy("/home/kist/바탕화면/storm/examples/data_ddp_0.3_30.txt")
# ddp_data = lmake_xy("/home/kist/바탕화면/storm/examples/data_ddp_0.001_90.txt")

data = data[:ddp_data.shape[0]]
pm_data = pm_data[:ddp_data.shape[0]]

print("d : ", data.shape, "pm_data: ",pm_data.shape, "ddp_data: ",ddp_data.shape)

print("storm x,y,z: ",data[0,0], ddp_data[0,1], ddp_data[0,2])
print("proposed x,y,z: ",pm_data[0,0], pm_data[0,1], pm_data[0,2])
print("proposed r,p,y: ",pm_data[0,3], pm_data[0,4], pm_data[0,5])
print("ddp x,y,z: ",ddp_data[0,0], ddp_data[0,1], ddp_data[0,2])
one = np.ones(data.shape[0])
# storm 좌표계가 무조코 꺼보다 y +=1.0 이고 축 방향이 반대임 (-)
data[:,1] = (data[:,1] - one)
data[:,7] = (data[:,7] - one)

temp1 = data[:,1].copy() # des_y
temp2 = data[:,7].copy() # ee_y

pm_data[:,2] = (pm_data[:,2] - 0.25*one)
pm_data[:,8] = (pm_data[:,8] - 0.25*one)

ddp_data[:,2] = (ddp_data[:,2] - 0.25*one)
ddp_data[:,8] = (ddp_data[:,8] - 0.25*one)

# storm y<->z축이 무조코랑 순서가 바뀌어서 맞게 열교환 해줌 + strom기준 z축 축 방향이 반대임 (-)
data[:,1] = -data[:,2]
data[:,2] = temp1

data[:,7] = -data[:,8]
data[:,8] = temp2

# storm y<->z축이 무조코랑 순서가 바뀌어서 맞게 열교환 해줌 + strom기준 z축 축 방향이 반대임 (-)
temp3 = data[:,4].copy() # des_pitch
temp4 = data[:,10].copy() # ee_pitch

data[:,4] = -data[:,5] #des_yaw
data[:,5] = temp3

data[:,10] = -data[:,11] #ee_yaw
data[:,11] = temp4

for i in range(pm_data.shape[0]):
        if (pm_data[i,9] <0.0):
                pm_data[i,9] = pm_data[i,9] + 2*np.pi
        if (ddp_data[i,9] < -1.5):
                ddp_data[i,9] = ddp_data[i,9] + 2*np.pi
        if(data[i,10] < -3.0):
                data[i,10] = data[i,10] + 2*np.pi
for i in range(21):
        data[540 + i,10] = data[539,10]
#         # data[4448 + i,10] = data[4448,10]
for i in range(21):
        data[4440 + i,10] = data[4439,10]

# des_pos = data[:,:3]
# des_rpy = data[:,3:6]
# ee_pos = data[:,6:9]
# ee_rpy = data[:,9:12]
compuation_time = data[:,12]
pm_compuation_time = pm_data[:,12]
ddp_compuation_time = ddp_data[:,12]
pm_compuation_time[0] = 0.2
print(pm_compuation_time.shape)

cp = np.mean(compuation_time)
cp_pm = np.mean(pm_compuation_time)
cp_ddp = np.mean(ddp_compuation_time)

Ac = 0
Bc = 0
for i in range(ddp_data.shape[0]):
        if ddp_data[i,6] >= 0.4999:
                if Ac <20:
                        print(i)
                        Ac = Ac +1
        if pm_data[i,8] >= 0.5999:
                if Bc <20:
                        print(i)
                        Bc = Bc +1
        


# print("storm mean time: ", cp * 1000, "ms", ",proposed mean time: ", np.mean(cp_pm) , "ms", ",fddp mean time: ", np.mean(cp_ddp)* 1000 , "ms")
print("storm mean time: ", cp * 1000, "ms", ",proposed mean time: ", np.mean(cp_pm) , "ms", ",fddp mean time: ", np.mean(cp_ddp)* 1000 , "ms")

mode = 1
if(mode == 1):
        pos = ["x","y","z", "roll", "pitch", "yaw"]

        reference_pos = data[:,:6] 
        current_pos = data[:,6:12]
        
        reference_pos_pm = pm_data[:,:6] 
        current_pos_pm = pm_data[:,6:12]
        
        reference_pos_fddp = ddp_data[:,:6] 
        current_pos_fddp = ddp_data[:,6:12]
        
        print(current_pos[0,2])
        print(current_pos_pm[0,2])
        print(current_pos[0,2]- current_pos_pm[0,2])

        # error = True
        error = False
        error_ = reference_pos-current_pos
        error_pm = reference_pos_pm-current_pos_pm
        error_ddp = reference_pos_fddp-current_pos_fddp
        
        # indices = np.where(current_pos_pm[:5] == 0.5)
        # indices_ddp = np.where(current_pos_fddp[:5] == 0.5)
        # print (indices)
        # print (indices_ddp)
        plt.figure(figsize=(10,7))
        for i in range(3):
                plt.subplot(3,1,i+1)

                # plt.plot(reference_pos[:,i+3], "--", label = "goal",linewidth="3")
                # plt.plot(current_pos[:,i+3],  label = "STORM",linewidth="3")
                            
                # plt.plot(reference_pos_pm[:,i+3], label = "Target_pm",linewidth="3")
                # plt.plot(current_pos_pm[:,i+3],  label = "Proposed method",linewidth="3")
                
                plt.plot(error_[:,i+3], label = "STORM",linewidth="3")
                plt.plot(error_pm[:,i+3],  label = "Proposed method",linewidth="3")
                # plt.plot(error_ddp[:,i+3],  label = "fddp",linewidth="3")
                            
                # plt.plot(reference_pos_fddp[:,i+3], label = "Target_fddp")
                # plt.plot(current_pos_fddp[:,i+3],  label = "current_fddp")
                if i==0:
                        plt.ylabel(" roll ($rad$)", fontsize=14)
                elif i==1:
                        plt.ylabel(" pitch ($rad$)", fontsize=14)
                if i==2:
                        plt.xlabel(" time ($s$) ", fontsize=14)
                        plt.ylabel(" yaw ($rad$)", fontsize=14)
                # plt.xticks([0, 2000, 4000, 6000, 8000, 10000],[0,2,4,6,8,10], fontsize=14)

                plt.yticks(fontsize=14)
                plt.subplots_adjust(
                wspace=0.25, # the width of the padding between subplots 
                hspace=0.3) #
                plt.grid()
                plt.legend(fontsize=14)
                plt.autoscale(enable=True, axis='x', tight=True)  # x축에 대해서만 타이트하게 조절
        # # plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_ori.pdf")
        # # plt.savefig("/home/kist/test/baseline_comparison_ori_29.svg")
        # plt.show()
        # plt.figure(figsize=(10,6))
        for i in range(3):
                plt.subplot(3,1,i+1)
                # plt.plot(reference_pos[:,i], "--", label = "goal", linewidth="3")
                # plt.plot(current_pos[:,i], label = "STORM", linewidth="3")
                            
                # # plt.plot(reference_pos_pm[:,i], label = "Target_pm")
                # plt.plot(current_pos_pm[:,i],  label = "Proposed method", linewidth="3")
                # plt.plot(current_pos_fddp[:,i], label = "fddp", linewidth="3")
                
                plt.plot(error_[:,i], label = "STORM",linewidth="3")
                plt.plot(error_pm[:,i],  label = "Proposed method",linewidth="3")
                # plt.plot(error_ddp[:,i],  label = "fddp",linewidth="3")
                            
                if i ==0:
                    plt.ylabel(" x ($m$)",fontsize=15)
                elif i== 1:
                    plt.ylabel(" y ($m$) ",fontsize=15)
                    plt.yticks([0.0, 0.05, 0.1],[0.0, 0.05, 0.1],fontsize=14)
                elif i== 2:
                    plt.xlabel(" time ($s$) ",fontsize=15)
                    plt.ylabel(" z ($m$) ",fontsize=15)
                            
                plt.xticks([0, 2000, 4000, 6000, 8000, 10000],[0,2,4,6,8,10])

                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.autoscale(enable=True, axis='x', tight=True)  # x축에 대해서만 타이트하게 조절
                plt.subplots_adjust(
                wspace=0.25, # the width of the padding between subplots 
                hspace=0.3) #
                plt.grid()
                plt.legend(fontsize=12)
        # image_name = '/home/kist/test/comparison_pos_ddp9.svg'
        # image_format = 'svg' # e.g .png, .svg, etc.
        # plt.savefig(image_name, format=image_format, dpi=1200)
        # plt.show()

        # plt.savefig("/home/kist/euncheol/mppi_test/baseline_comparison_pos.pdf")
        # plt.show()
        
        
        
        # plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_ori.svg")
        # plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_ori.pdf")
        # plt.figure(figsize=(20,10))
        # if(error):
        #         for i in range(6):
        #                 plt.subplot(6,1,i+1)
        #                 plt.title("position {}".format([pos[i]]))
        #                 plt.plot(error_[:,i], label = "error")
        #                 plt.xlabel(" time (ms) ")
        #                 if i <3:
        #                         plt.ylabel(" distance (m)")
        #                 else:
        #                         plt.ylabel(" angle (rad)")
        #                 plt.subplots_adjust(
        #                 wspace=0.25, # the width of the padding between subplots 
        #                 hspace=0.3) #
                        
        #                 plt.legend()
                
        # else:
        #     for num in range(2):
        #         if num == 1:
        #             for i in range(3):
        #                     plt.subplot(3,1,i+1)
        #                     plt.title("position {}".format([pos[i]]))

        #                     plt.plot(reference_pos[:,i], linewidth = "3",label = "Target")
        #                     plt.plot(current_pos[:,i],  label = "current_storm")
                            
        #                     # plt.plot(reference_pos_pm[:,i], label = "Target_pm")
        #                     plt.plot(current_pos_pm[:,i],  label = "current_pm")
                            
        #                     # plt.plot(reference_pos_fddp[:,i], label = "Target_fddp")
        #                     plt.plot(current_pos_fddp[:,i],  label = "current_fddp")

        #                     plt.xlabel(" time ($s$) ")
        #                     plt.xticks([0, 2000, 4000, 6000, 8000, 10000],[0,2,4,6,8,10])

        #                     plt.ylabel(" distance ($m$)")

        #                     plt.subplots_adjust(
        #                     wspace=0.25, # the width of the padding between subplots 
        #                     hspace=0.3) #
        #                     plt.grid()
        #                     plt.legend()
        #             plt.show()
        #             plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_ori.svg")
        #             plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_ori.pdf")
        #         # else:
        #         #     for i in range(3):
        #         #             plt.subplot(3,1,i+1)
        #         #             plt.title("Orientation {}".format([pos[i+i]]))

        #         #             plt.plot(reference_pos[:,i+3], label = "Target_storm")
        #         #             plt.plot(current_pos[:,i+3],  label = "current_storm")
                            
        #         #             plt.plot(reference_pos_pm[:,i+3], label = "Target_pm")
        #         #             plt.plot(current_pos_pm[:,i+3],  label = "current_pm")
                            
        #         #             plt.plot(reference_pos_fddp[:,i+3], label = "Target_fddp")
        #         #             plt.plot(current_pos_fddp[:,i+3],  label = "current_fddp")

        #         #             plt.xlabel(" time ($s$) ")
        #         #             plt.xticks([0, 2000, 4000, 6000, 8000, 10000],[0,2,4,6,8,10])

        #         #             plt.ylabel(" angle (rad)")
        #         #             plt.subplots_adjust(
        #         #             wspace=0.25, # the width of the padding between subplots 
        #         #             hspace=0.3) #
        #         #             plt.grid()
        #         #             plt.legend()
        #         #     plt.show()
        #         #     plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_pos.svg")
        #         #     plt.savefig("/home/kist/euncheol/mppi_test/baselin_comparison_pos.pdf")
                    
else:
        print("none~~~")
        
        
        
        
#################
# # 두 알고리즘의 연산 시간 데이터
# # cp_pm = np.random.normal(loc=2, scale=0.5, size=10000)
# # cp_ddp = np.random.normal(loc=2.5, scale=0.5, size=10000)

# ddp_compuation_time = ddp_compuation_time*1000
     
# cp_pm = np.squeeze(pm_compuation_time)
# cp_ddp = np.squeeze(ddp_compuation_time)

# print("pm_compuation_time: ",pm_compuation_time)
# print("ddp_compuation_time: ",ddp_compuation_time)
# print(np.mean(pm_compuation_time), np.mean(ddp_compuation_time))

# data = [cp_pm, cp_ddp]  # 데이터를 리스트로 정리
# plt.boxplot(data, notch=True, patch_artist=True, labels=['Proposed', 'FDDP'])  # 올바른 변수 이름 사용
# plt.ylabel('Computation Time (ms)')
# plt.title('Comparison of Computation Times')
# # image_name = '/home/kist/test/comparison_pos_ddp_box_lib.png'
# # image_format = 'png' # e.g .png, .svg, etc.
# # plt.savefig(image_name, format=image_format, dpi=1200)
# plt.show()


# # # 박스플롯 생성
# # plt.figure(figsize=(6, 4))  # 그래프 크기 설정
# # plt.boxplot(data, notch=True, patch_artist=True, labels=['Proposed', 'FDDP'], boxprops=dict(facecolor='lightblue'))
# # plt.ylabel('Computation Time (s)')  # Y축 레이블
# # plt.title('Comparison of Computation Times')
# # plt.grid(True)  # 격자 표시
# # plt.show()

# plt.violinplot(data)
# plt.xticks([1, 2], ['Proposed_SimOff', 'FDDP'])
# plt.ylabel('Computation Time (ms)')
# plt.title('Comparison of Computation Times')
# # image_name = '/home/kist/test/comparison_pos_ddp_violine_lib.png'
# # image_format = 'png' # e.g .png, .svg, etc.
# # plt.savefig(image_name, format=image_format, dpi=1200)
# plt.show()       
        
# # plt.hist(data)        
# # plt.hist(data, bins = 30)
# # plt.show()       

# # from scipy.stats import gaussian_kde
# # density = gaussian_kde(cp_pm)
# # xs = np.linspace(min(cp_pm), max(cp_pm), 1000)
# # density1 = gaussian_kde(cp_ddp)
# # xs1 = np.linspace(min(cp_ddp), max(cp_ddp), 1000)
# # plt.plot(xs, density(xs))
# # plt.plot(xs1, density1(xs1))
# # plt.title('KDE Plot')
# # plt.show()
        