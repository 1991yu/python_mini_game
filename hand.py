import cv2
import mediapipe as mp
import numpy as np
import random
import math
from PIL import Image
from playsound import playsound
from pygame import mixer 
import time 

# mixer.init()
# mixer.music.load('xx.mp3')
# mixer.music.play()
# time.sleep(5)
# mixer.music.stop()


#初始化部分
# model = True
aim_x, aim_y = 0, 0
aim_x2, aim_y2 = 0, 0
aim_model = True #初始化模式为创造
aim_model2 = True #初始化模式为创造
score=0


#定义游戏模式
gamemodel = True  #初始化为界面模式
gamelife = True #初始化为存活模式
life = 1

#定义阶段变量
step = 0
step2 = 0
boom_model1 = False
boom_model2 = False

gameStop = False


# 定义了一个手部识别的工具
class HandDetector():
    def __init__(self):
        #手势识别器，读取的是mediapipe的手部识别解决方案
        self.hand_detector  = mp.solutions.hands.Hands()
        
        self.drawer = mp.solutions.drawing_utils

    def process(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #将图片从BGR格式变成RGB格式
        self.hands_data = self.hand_detector.process(img_rgb)
        #print(result.multi_hand_landmarks)  

        if self.hands_data.multi_hand_landmarks:  # 如果有获取到手势数据的话
            for handlms in self.hands_data.multi_hand_landmarks:
                #画出
                self.drawer.draw_landmarks(img, handlms, mp.solutions.hands.HAND_CONNECTIONS)
        else:
            gameStop = True
        

    def find_position(self, img):  #获取识别是左手还是右手
        h, w, c = img.shape
        #print(h,w,c)
        position = {'Left':{},'Right':{}}
        if self.hands_data.multi_hand_landmarks:
            for i, point in enumerate(self.hands_data.multi_handedness):

                score = point.classification[0].score

                if score >=0.8:#如果准确率大于80%

                    label = point.classification[0].label
                    hand_lms = self.hands_data.multi_hand_landmarks[i].landmark

                    for id , lm in enumerate(hand_lms):
                        x, y = int(lm.x * w), int(lm.y * h)
                        position[label][id] = (x, y)
        return position

    # def hand_stop():
    #     r=pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)
        

    # def fingerStatus(self, lmList):

    #     fingerList = []
    #     id, originx, originy = lmList[0]
    #     keypoint_list = [[2, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
    #     for point in keypoint_list:
    #         id, x1, y1 = lmList[point[0]]
    #         id, x2, y2 = lmList[point[1]]
    #         if math.hypot(x2-originx, y2-originy) > math.hypot(x1-originx, y1-originy):
    #             fingerList.append(True)
    #         else:
    #             fingerList.append(False)

    #     return fingerList





# 图片叠加功能模块
def picplus(img1,img2,coordinate):
    img1_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)) #转换为PIL格式
    img2_pil = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    img1_pil.paste(img2_pil, coordinate) #img2贴在img1指定位置，位置是(左,上)
    return cv2.cvtColor(np.asarray(img1_pil), cv2.COLOR_RGB2BGR)


#图片显示功能模块
def image_plus(img_back,img,center):

    # img=cv2.imread('zhadan.jpg')
    # img_back=cv2.imread('sky.jpg')
    #日常缩放

    rows,cols,channels = img_back.shape
    img_back=cv2.resize(img_back,None,fx=1,fy=1)
    #cv2.imshow('img_back',img_back)

    rows,cols,channels = img.shape
    img=cv2.resize(img,None,fx=1,fy=1)

    #cv2.imshow('img',img)
    rows,cols,channels = img.shape#rows，cols最后一定要是前景图片的，后面遍历图片需要用到

    #转换hsv
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #获取mask
    lower_blue=np.array([78,43,46])
    upper_blue=np.array([110,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #cv2.imshow('Mask', mask)

    #腐蚀膨胀
    erode=cv2.erode(mask,None,iterations=1)
    #cv2.imshow('erode',erode)

    dilate=cv2.dilate(erode,None,iterations=1)
    #cv2.imshow('dilate',dilate)

    #遍历替换
    #center=[50,50]#在新背景图片中的位置
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==0:#0代表黑色的点
                img_back[center[0]+i,center[1]+j]=img[i,j]#此处替换颜色，为BGR通道
    
    #cv2.imshow('res',img_back)

    return(img_back)



#两点间距离计算模块
def distance(x1,y1,x0,y0):
    r=pow(pow(x1-x0,2)+pow(y1-y0,2),0.5)
    return r






def Boom(img,step,aim_x,aim_y):
    #显示爆炸动画1
    if step == 1:
        img = image_plus(img,boom1,(aim_x-15, aim_y-15))
        return img
    #显示爆炸动画2
    if step == 2:
        img = image_plus(img,boom2,(aim_x-15, aim_y-15))
        return img
    #显示爆炸动画3
    if step == 3:
        img = image_plus(img,boom3,(aim_x-15, aim_y-15))
        return img
    #显示爆炸动画4
    if step == 4:
        img = image_plus(img,boom4,(aim_x-15, aim_y-15))
        return img
    



#摄像头读取函数
camera = cv2.VideoCapture(0)
#定义识别手部的对象
hand_detector  = HandDetector()




#图片读取以及调整大小
TNT_B = cv2.imread('TNT_B.jpg')
x, y = TNT_B.shape[:2]
TNT_B = cv2.resize(TNT_B, (int(y / 25), int(x / 25)))
#daodan = cv2.flip(img2,0)

TNT_G = cv2.imread('TNT_G.jpg')
x, y = TNT_G.shape[:2]
TNT_G = cv2.resize(TNT_G, (int(y / 25), int(x / 25)))
#daodan = cv2.flip(img2,0)

#爆炸动画1
boom1 = cv2.imread('boom1.jpg')
x, y = boom1.shape[:2]
boom1 = cv2.resize(boom1, (int(y / 2), int(x / 2)))

#爆炸动画2
boom2 = cv2.imread('boom2.jpg')
x, y = boom2.shape[:2]
boom2 = cv2.resize(boom2, (int(y / 2), int(x / 2)))

#爆炸动画3
boom3 = cv2.imread('boom3.jpg')
x, y = boom3.shape[:2]
boom3 = cv2.resize(boom3, (int(y / 2), int(x / 2)))

#爆炸动画4
boom4 = cv2.imread('boom4.jpg')
x, y = boom4.shape[:2]
boom4 = cv2.resize(boom4, (int(y / 2), int(x / 2)))

#读取城市的图片
city = cv2.imread('city.jpg')
x, y = city.shape[:2]
city = cv2.resize(city, (int(y / 0.8), int(x / 2.5)))





#碰撞检测函数
def collision(finger_y,finger_x,aim_y,aim_x):
    game_model = 15
    if aim_x - game_model <= finger_x <= aim_x + game_model and aim_y - game_model <= finger_y <= aim_y + game_model:
        return True
        

#开始循环
while True:

    success, img = camera.read()  # 返回两个值，第一个是布尔类型反应有没有读取成功，img是每一帧的图片
    #h1, w1, c1 = img.shape
    if success:

        img = cv2.flip(img, 1) # 将画面翻转

        hand_detector.process(img)
        position = hand_detector.find_position(img)
        gameStop = not position['Left'] and not position['Right']
        #img = Picture_Synthesis(img,daodan,(300,300))
        #img = pic(daodan,img,(300,300))

        #imgae_y = imgae_y + 3
        #img = picplus(img,img2,(imgae_x,imgae_y))
        #img = image_plus(img,img2,(imgae_y,imgae_x))

        #菜单模式
        if gamemodel:  

            #画出两个开始按钮的点            
            cv2.circle(img, (320, 200),10,(255, 255, 255),cv2.FILLED)
            cv2.circle(img, (320, 300),10,(255, 255, 255),cv2.FILLED)

            #画出推出按钮
            text2 = 'Exit'
            cv2.putText(img,text2,(330,300),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)

            #判断是否为第一次死亡
            if gamelife:
                starts = 'Start'
                text1 = starts
                cv2.putText(img,text1,(330,200),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
                text2 = 'AimGame'
                cv2.putText(img,text2,(200,50),cv2.FONT_HERSHEY_PLAIN,4.0,(255,255,255),2)

            #如果是第二次死亡则会显示重新开始和游戏结束
            else:
                starts = 'Start again'
                text1 = starts
                cv2.putText(img,text1,(330,200),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)
                cv2.putText(img,'GameOver',(180,50),cv2.FONT_HERSHEY_PLAIN,4.0,(255,255,255),2)

            #重新定义生命值，初始化生命值
            life = 3

            #检测左手有没有碰到按钮
            if left_finger := position['Left'].get(8, None):
                #print(left_finger)
                #画出手指尖端的小球
                cv2.circle(img, (left_finger[0], left_finger[1]),10,(255, 0, 0),cv2.FILLED)

                if collision(left_finger[0],left_finger[1],320,200):
                    if not gamelife:
                        #重新格式化分数
                        score = 0  

                    #更改游戏模式
                    gamemodel = False
                    gamelife = True

                #检测是否碰撞到退出界面
                if collision(left_finger[0],left_finger[1],320,300):  
                    break

            #检测右手有没有碰到按钮
            if right_finger := position['Right'].get(8, None):
                #画出手指尖端的小球
                cv2.circle(img, (right_finger[0], right_finger[1]),10,(0, 255, 0),cv2.FILLED)

                if collision(right_finger[0],right_finger[1],320,200):  
                    if not gamelife:
                        #重新格式化分数
                        score = 0  

                    #更改游戏模式
                    gamemodel = False
                    gamelife = True
                
                #检测是否碰撞到退出界面
                if collision(right_finger[0],right_finger[1],320,300):  
                    break


        elif gamelife:  #如果gamelife为真，那么就会进行游戏模式
            #通过检测aim的模式有没有改变为创造来判断，如果改变为创造了就创造一个新的小球，并将模式改变为存活  480 640 
            # left_hand = position['Left'].get(0, None)
            # right_hand = position['Right'].get(0, None)
            # if left_hand[0] == None  and  left_hand[1] == None:
            #     print('stop')
            #     gameStop = True
            # print(left_finger[0],right_finger[1],left_hand[0],right_hand[1],distance(left_hand[0],left_hand[1],left_finger[0],left_finger[1]),distance(right_hand[0],right_hand[1],right_finger[0],right_finger[1]))

            # #通过左右手的握合来控制游戏的暂停和继续
            # #左手部分

            #     #print(distance(left_hand[0],left_hand[1],left_finger[0],left_finger[1]))

            # if distance(left_hand[0],left_hand[1],left_finger[0],left_finger[1]) <= 80 or distance(right_hand[0],right_hand[1],right_finger[0],right_finger[1]) <= 80:
            #     gameStop = True
            # else:
            #     gameStop = False



            #小球的随机生产部分
            if aim_model == True:
                aim_x=random.randint(0,20)
                aim_y=random.randint(50,300)  #可以更改小球的刷新位置
                aim_model = False #将小球的生存模式变成存活状态

            if aim_model2 == True:
                aim_x2=random.randint(0,20)
                aim_y2=random.randint(300,550)
                aim_model2 = False #将小球的生存模式变成存活状态


            #检测小球的状态是否为存活，如果存活要逐渐下降
            #控制小球1的下落速度
            if aim_model == False:
                #如果游戏没有暂停

                if not gameStop:

                    #根据得分的变化来逐渐提高炸弹下落的速度来提升难度
                    if score <= 2:
                        aim_x = aim_x + 2
                    elif 3<= score <5:
                        aim_x = aim_x + 4
                    elif 5 <= score < 10:
                        aim_x = aim_x + 6
                    elif 10 <= score < 15:
                        aim_x = aim_x + 8
                    else :
                        aim_x = aim_x + 10

                #超过400个像素点就视为没有接住小球，就会减少血量，并重新刷新小球
                if aim_x > 400:
                    life = life - 1
                    aim_model = True  

            #控制小球2的下落速度
            if aim_model2 == False:
                #如果游戏没有暂停

                if not gameStop:
                    #根据得分的变化来逐渐提高炸弹下落的速度来提升难度
                    if score <= 2:
                        aim_x2 = aim_x2+ 2
                    elif 3<= score <5:
                        aim_x2 = aim_x2 + 4
                    elif 5 <= score < 10:
                        aim_x2 = aim_x2 + 6
                    elif 10 <= score < 15:
                        aim_x2 = aim_x2 + 8
                    else :
                        aim_x2 = aim_x2 + 10

                #超过400个像素点就视为没有接住小球，就会减少血量，并重新刷新小球
                if aim_x2 > 400:
                    life = life - 1
                    aim_model2 = True  

            #画面显示部分
            #显示爆炸的判断部分

            img = picplus(img,city,(0,400))

            if boom_model1:
                if step == 5:
                    boom_model1 = False
                else:
                    img = Boom(img,step,boom_x,boom_y)
                    step = step + 1

            if boom_model2:
                if step2 == 5:
                    boom_model2 = False
                else:
                    img = Boom(img,step2,boom_x2,boom_y2)
                    step2 = step2 + 1


            #显示游戏的血量
            text_life = str(life)
            cv2.putText(img,text_life,(400,50),cv2.FONT_HERSHEY_PLAIN,2.0,(50,50,205),2)


            #显示游戏的得分
            text = str(score)
            cv2.putText(img,text,(280,50),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2)

            #分别显示两个小球

            #显示小球1
            img = image_plus(img,TNT_B,(aim_x-15, aim_y-15))
            #cv2.circle(img, (aim_y, aim_x),10,(255, 0, 0),cv2.FILLED)

            #显示小球2
            img = image_plus(img,TNT_G,(aim_x2-15, aim_y2-15))
            #cv2.circle(img, (aim_y2, aim_x2),10,(0, 255, 0),cv2.FILLED)


            #检测左手有没有碰到小球
            if left_finger := position['Left'].get(8, None):
                #print(left_finger)
                #将手指指尖标记出来
                cv2.circle(img, (left_finger[0], left_finger[1]),10,(255, 0, 0),cv2.FILLED)

                if collision(left_finger[0],left_finger[1],aim_y,aim_x):  # 如果检测到碰撞就会改变aim的模式为创造
                    #cv2.circle(img, (aim_y, aim_x),10,(100, 255, 255),cv2.FILLED)
                    #如果碰到就将爆炸模式改为真，然后程序会继续进行爆炸模式，完成爆炸就会显示为假
                    boom_model1 = True

                    #给爆炸坐标
                    boom_x = aim_x
                    boom_y = aim_y
                    step = 1

                    #增加得分
                    score += 1

                    #print(score)
                    aim_model = True


            #检测右手有没有碰到小球
            if right_finger := position['Right'].get(8, None):
                #将手指指尖标记出来
                cv2.circle(img, (right_finger[0], right_finger[1]),10,(0, 255, 0),cv2.FILLED)

                if collision(right_finger[0],right_finger[1],aim_y2,aim_x2):  # 如果检测到碰撞就会改变aim的模式为创造
                    #cv2.circle(img, (aim_y2, aim_x2),10,(0, 100, 0),cv2.FILLED)

                    #如果碰到就将爆炸模式改为真，然后程序会继续进行爆炸模式，完成爆炸就会显示为假
                    boom_model2 = True

                    #给爆炸坐标
                    boom_x2 = aim_x2
                    boom_y2 = aim_y2
                    step2 = 1

                    #增加得分
                    score += 1
                    #print(score)
                    aim_model2 = True


            #手部判断代码，握拳则退出为开始界面
            if left_hand := position['Left'].get(0, None):
                #判断手部是否握拳，只对左手有效
                if left_hand[0] and left_hand[1] and left_finger[0] and left_finger[1] and distance(left_hand[0], left_hand[1], left_finger[0], left_finger[1]) <= 80:
                    gamelife = False
                    gamemodel = True

            #如果生命值减少到0，则更改游戏模式到菜单界面
            if life < 1:
                gamelife = False
                gamemodel = True


        #print(position)
        #hand_detector.find_positions(img) #左手还是右手
        
        #显示整个画面
        cv2.imshow('video',img)

    #等待按键，等待的是1ms
    k = cv2.waitKey(1)

    # 如果k=q按键
    if k == ord('q'):  
        break

#释放掉摄像头，并销毁窗口
camera.release()
cv2.destroyAllWindows()



# def pic(bg,fg):  #dst是背景图

#     dim = (35,70)
#     resized_bg=cv2.resize(bg,dim,interpolation=cv2.INTER_AREA)
#     resized_fg=cv2.resize(fg,dim,interpolation=cv2.INTER_AREA)

#     return cv2.addWeighted(resized_bg,0.5,resized_fg,0.8,0.0)



# def Picture_Synthesis(mother_img,
#                       son_img,
#                       coordinate):
#     """
#     :param mother_img: 母图
#     :param son_img: 子图
#     :param save_img: 保存图片名
#     :param coordinate: 子图在母图的坐标
#     :return:
#     """
#     #将图片赋值,方便后面的代码调用
#     M_Img = mother_img
#     S_Img = son_img
#     factor = 1#子图缩小的倍数1代表不变，2就代表原来的一半

#     #给图片指定色彩显示格式
#     M_Img = M_Img.convert("RGBA")  # CMYK/RGBA 转换颜色格式（CMYK用于打印机的色彩，RGBA用于显示器的色彩）

#     # 获取图片的尺寸
#     M_Img_w, M_Img_h = M_Img.size  # 获取被放图片的大小（母图）
#     print("母图尺寸：",M_Img.size)
#     S_Img_w, S_Img_h = S_Img.size  # 获取小图的大小（子图）
#     print("子图尺寸：",S_Img.size)

#     size_w = int(S_Img_w / factor)
#     size_h = int(S_Img_h / factor)

#     # 防止子图尺寸大于母图
#     S_Img_w = min(S_Img_w, size_w)
#     S_Img_h = min(S_Img_h, size_h)

#     # # 重新设置子图的尺寸
#     # icon = S_Img.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)
#     icon = S_Img.resize((S_Img_w, S_Img_h), Image.ANTIALIAS)
#     w = int((M_Img_w - S_Img_w) / 2)
#     h = int((M_Img_h - S_Img_h) / 2)

#     try:
#         if coordinate is None or coordinate == "":
#             coordinate=(w, h)
#         else:
#             print("已经指定坐标")
#         # 粘贴子图到母图的指定坐标（当前居中）
#         M_Img.paste(icon, coordinate, mask=None)
#     except Exception:
#         print("坐标指定出错 ")
#     # 保存图片
#     #M_Img.save(save_img)
#     return M_Img