"""Static UI proposal only. Does not import or modify ImageLab application code."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
W, H = 1920, 1200
S = 2
BG = '#11161d'
PANEL = '#1a222d'
FIELD = '#111923'
LINE = '#303d4d'
TEXT = '#e7edf5'
MUTED = '#93a3b7'
ACCENT = '#79b9ff'
GREEN = '#69d3ae'
AMBER = '#efbe73'
FONT = 'C:/Windows/Fonts/msyh.ttc'
BOLD = 'C:/Windows/Fonts/msyhbd.ttc'


class Canvas:
    def __init__(self):
        self.im = Image.new('RGB', (W*S,H*S), BG)
        self.d = ImageDraw.Draw(self.im)

    def rect(self, box, fill, outline=None, radius=0, width=1):
        coords = tuple(round(v*S) for v in box)
        if radius:
            self.d.rounded_rectangle(coords, radius=radius*S, fill=fill, outline=outline, width=width*S)
        else:
            self.d.rectangle(coords, fill=fill, outline=outline, width=width*S)

    def text(self, x, y, value, size=20, color=TEXT, bold=False, anchor=None):
        self.d.text((x*S,y*S), value, font=ImageFont.truetype(BOLD if bold else FONT, size*S), fill=color, anchor=anchor)

    def line(self,x,y,x2,y2,color=LINE,width=1):
        self.d.line((x*S,y*S,x2*S,y2*S), fill=color,width=width*S)

    def button(self,x,y,w,label,kind='normal',h=42):
        fill,fg,edge = {'normal':('#263242',TEXT,LINE),'primary':('#3376bc','#ffffff','#478acb'),
                        'green':('#236d59','#f2fff9','#37866d'),'active':('#233d58',ACCENT,'#4a78a2'),
                        'disabled':('#202934','#657487',LINE),'ghost':(PANEL,MUTED,PANEL)}[kind]
        self.rect((x,y,x+w,y+h),fill,edge,7)
        self.text(x+w/2,y+h/2,label,18,fg,kind in ('primary','green'),anchor='mm')

    def field(self,x,y,w,value,arrow=False,h=40,active=False):
        self.rect((x,y,x+w,y+h),FIELD,ACCENT if active else LINE,6)
        self.text(x+12,y+9,value,18)
        if arrow:
            cx,cy=x+w-20,y+h/2
            self.line(cx-5,cy-2,cx,cy+3,MUTED,2)
            self.line(cx,cy+3,cx+5,cy-2,MUTED,2)

    def check(self,x,y,label,checked=True,disabled=False):
        fg = '#627184' if disabled else TEXT
        self.rect((x,y+3,x+18,y+21),'#286cad' if checked and not disabled else FIELD,LINE,3)
        if checked:
            tick='white' if not disabled else '#627184'
            self.line(x+4,y+12,x+8,y+16,tick,2)
            self.line(x+8,y+16,x+14,y+8,tick,2)
        self.text(x+29,y,label,18,fg)

    def tag(self,x,y,label,color=GREEN):
        n = len(label)*17+22
        self.rect((x,y,x+n,y+30),'#21312f' if color==GREEN else '#392f23',None,5)
        self.text(x+11,y+4,label,16,color)

    def section(self,x,y,title,sub=None):
        self.text(x,y,title,21,TEXT,True)
        if sub:self.text(x,y+34,sub,17,MUTED)

    def progress(self,x,y,w,percent,label):
        self.text(x,y,label,17,MUTED)
        self.text(x+w,y,f'{percent}%',17,GREEN,anchor='ra')
        self.rect((x,y+31,x+w,y+37),'#2b3847',None,3)
        self.rect((x,y+31,x+w*max(.002,percent/100),y+37),GREEN,None,3)

    def paste_fit(self,image,box,padding=0):
        x,y,x2,y2=box
        image=image.copy().convert('RGB')
        image.thumbnail((int((x2-x-padding*2)*S),int((y2-y-padding*2)*S)),Image.Resampling.LANCZOS)
        px=int(x*S+(x2-x)*S/2-image.width/2)
        py=int(y*S+(y2-y)*S/2-image.height/2)
        self.im.paste(image,(px,py))
        return (px/S,py/S,image.width/S,image.height/S)

    def save(self,name):
        self.im.resize((W,H),Image.Resampling.LANCZOS).save(OUT/name)


def header(c,tab=0):
    c.rect((0,0,W,78),'#17202b')
    c.rect((24,19,62,57),'#3376bc',None,9)
    c.text(43,38,'I',24,'white',True,'mm')
    c.text(76,15,'ImageLab',27,TEXT,True)
    c.text(77,47,'文献图像工作台',14,MUTED)
    for i,label in enumerate(('01  整页去背景','02  单字提取')):
        x=355+i*228
        if i==tab:c.rect((x,14,x+208,64),'#25384e',None,7)
        c.text(x+104,39,label,21,ACCENT if i==tab else MUTED,i==tab,'mm')
        if i==tab:c.rect((x+50,73,x+158,77),ACCENT)
    c.text(1470,30,'界面设计示意 · 待确认',17,MUTED)
    c.button(1758,19,136,'任务中心',h=40)
    c.rect((24,96,1896,154),PANEL,LINE,8)
    c.text(40,114,'原图路径',18,MUTED)
    c.field(132,104,658,r'D:\…\谢福平金刚经-600')
    c.button(800,104,74,'选择',h=40)
    c.text(908,114,'导出路径',18,MUTED)
    c.field(1000,104,786,r'D:\…\谢福平金刚经-600\导出')
    c.button(1796,104,78,'选择',h=40)


def panels(c,tab=0):
    header(c,tab)
    c.rect((24,174,1456,1158),PANEL,LINE,10)
    c.rect((1474,174,1896,1158),PANEL,LINE,10)
    c.text(26,1175,'设计预览：展示拟议交互，文字识别与状态为示意，不代表已实现功能。',16,MUTED)


def pager(c,title='原图与去背景结果对比'):
    c.text(44,194,title,22,TEXT,True)
    c.button(340,187,104,'重置视图',h=38)
    c.button(780,187,100,'上一张',h=38)
    c.button(896,187,100,'下一张',h=38)
    c.field(1010,187,110,'1 / 371',h=38)
    c.text(1144,196,'跳至',18,MUTED)
    c.field(1193,187,100,'1',h=38)
    c.button(1306,187,128,'转到',h=38)


source=Image.open(ROOT/'谢福平金刚经/谢福平金刚经-600/字帖001.jpg')
source.thumbnail((1600,1600),Image.Resampling.LANCZOS)
# Illustrative transparent-text content for UI review, not a processing algorithm change.
arr=np.array(source)
alpha=(arr[:,:,1] < 100).astype('uint8')*255
clean=source.convert('RGBA');clean.putalpha(Image.fromarray(alpha))
white=Image.new('RGB',source.size,'white');white.paste(clean,mask=clean.getchannel('A'))
glyphs=['住','法','所','語','法','而','得','者','而','行','法','如']
# Normalized bounds on the actual sample page, in horizontal reading order.
bounds=[(.060,.115,.250,.350),(.305,.130,.488,.350),(.540,.140,.728,.370),(.775,.115,.982,.355),
        (.058,.385,.253,.632),(.306,.425,.490,.626),(.540,.430,.734,.640),(.776,.425,.982,.660),
        (.066,.708,.248,.902),(.288,.707,.493,.932),(.535,.704,.735,.919),(.779,.716,.985,.950)]


def glyph_image(i):
    b=bounds[i]
    return clean.crop(tuple(round(v*(clean.width if j%2==0 else clean.height)) for j,v in enumerate(b)))


def checker(c,box,cell=12):
    x,y,x2,y2=map(int,box)
    c.rect(box,'#f7f8fa')
    for yy in range(y,y2,cell):
        for xx in range(x,x2,cell):
            if ((xx-x)//cell+(yy-y)//cell)%2==0:c.rect((xx,yy,min(x2,xx+cell),min(y2,yy+cell)),'#e5e9ee')


def transparent_fit(c,img,box,pad=10):
    checker(c,box)
    im=img.copy()
    im.thumbnail((int((box[2]-box[0]-2*pad)*S),int((box[3]-box[1]-2*pad)*S)),Image.Resampling.LANCZOS)
    c.im.paste(im,(int((box[0]+box[2])*S/2-im.width/2),int((box[1]+box[3])*S/2-im.height/2)),im)


def page_screen():
    c=Canvas();panels(c);pager(c)
    c.text(44,246,'字帖001.jpg',19,TEXT,True)
    c.tag(213,243,'已有处理结果')
    c.text(389,248,'翻页只查看；点击“处理当前图片”才重新处理',17,MUTED)
    for x,title,img in ((44,'原图',source),(752,'去背景结果 · 透明区域以白色预览',white)):
        c.text(x,300,title,19,TEXT,True)
        c.rect((x,342,x+680,991),'#111820',LINE,7)
        c.paste_fit(img,(x+15,357,x+665,975),10)
        c.text(x+16,1010,'600 dpi  ·  9573 × 7247 px',17,MUTED)
    c.line(44,1050,1434,1050)
    c.text(44,1068,'Alt + 滚轮缩放   /   空格 + 拖动平移   /   两侧视图同步',18,MUTED)
    c.text(44,1105,'结果窗口：Ctrl + 左键擦除 · Ctrl + 滚轮调大小 · Ctrl + Z 撤销',17,MUTED)
    c.button(1163,1080,270,'前往单字提取 →','primary',46)
    x=1496;fw=378
    c.section(x,196,'去背景设置')
    c.text(x,241,'处理模板',17,MUTED)
    c.field(x,269,fw,'古籍 · 平衡保留墨色',True)
    c.button(x,321,181,'保存模板');c.button(x+193,321,185,'另存为')
    for y,label,value in ((388,'资料类型','古籍扫描'),(442,'去背景策略','平衡去背景'),(496,'文字判定偏移','3')):
        c.text(x,y+9,label,18,MUTED);c.field(x+154,y,224,value,y!=496)
    c.check(x,553,'保留淡色笔画')
    c.button(x,594,fw,'高级分析参数 · 展开',h=38)
    c.line(x,655,x+fw,655)
    c.section(x,677,'整页导出')
    c.text(x,722,'输出格式',18,MUTED);c.field(x+154,712,224,'TIF',True)
    c.check(x,771,'添加最底部白底图层',False)
    c.check(x,814,'统一输出分辨率',False)
    c.field(x+233,805,145,'300 dpi',h=38)
    c.text(x,858,'未勾选时保留原图尺寸和 DPI',17,MUTED)
    c.button(x,900,181,'处理当前图片','primary',46)
    c.button(x+193,900,185,'保存当前图片','green',46)
    c.button(x,960,181,'自动批量处理',h=42)
    c.button(x+193,960,185,'取消批量','disabled',42)
    c.progress(x,1025,fw,100,'当前图片处理完成 · 耗时 8.42 秒')
    c.text(x,1094,'文字 RGB 保留原值，仅移除背景',17,GREEN)
    c.save('01-page-cleanup.png')


def character_screen():
    c=Canvas();panels(c,1);pager(c,'单字区域与顺序校对')
    c.text(44,246,'字帖001.jpg',19,TEXT,True)
    c.tag(213,243,'12 个候选字')
    c.text(386,247,'横排 · 从左到右逐行  /  已确认 10 · 待校对 2',17,MUTED)
    for x,w,label,kind in ((44,92,'选择','active'),(146,130,'调整边界','normal'),(286,94,'合并','normal'),(390,94,'拆分','normal'),(494,120,'补充字框','normal'),(624,94,'删除','normal')):
        c.button(x,294,w,label,kind,38)
    c.text(1000,303,'显示：原图 / 去背景结果',17,MUTED)
    c.rect((44,350,1088,903),'#111820',LINE,7)
    rect=c.paste_fit(white,(60,360,1072,892),8)
    px,py,pw,ph=rect
    for i,b in enumerate(bounds):
        x1,y1,x2,y2=px+b[0]*pw,py+b[1]*ph,px+b[2]*pw,py+b[3]*ph
        color=ACCENT if i==0 else (AMBER if i in (3,10) else GREEN)
        c.rect((x1-4,y1-4,x2+4,y2+4),None,color,3,2)
        c.rect((x1-4,y1-29,x1+45,y1-3),color,None,4)
        c.text(x1+21,y1-16,f'{i+1:03}',14,BG,True,'mm')
        if i==0:
            for xx,yy in ((x1-4,y1-4),(x2+4,y1-4),(x1-4,y2+4),(x2+4,y2+4)):
                c.rect((xx-3,yy-3,xx+3,yy+3),TEXT,ACCENT)
    c.text(1110,357,'选中字预览',18,TEXT,True)
    transparent_fit(c,glyph_image(0),(1110,399,1434,725),24)
    c.text(1110,749,'001  ·  住',24,TEXT,True)
    c.tag(1110,794,'已确认')
    c.text(1110,849,'预览底色：透明棋盘格',16,MUTED)
    c.text(44,929,'单字列表',20,TEXT,True)
    c.text(159,934,'全部 12    已确认 10    待校对 2',17,MUTED)
    c.text(802,930,'显示 1–6 / 12',17,MUTED)
    c.button(975,920,56,'‹',h=36)
    c.button(1040,920,56,'›',h=36)
    c.button(1250,920,184,'只看待校对',h=36)
    for i in range(6):
        x=44+i*235
        c.rect((x,973,x+216,1134),FIELD,ACCENT if i==0 else LINE,7,2 if i==0 else 1)
        transparent_fit(c,glyph_image(i),(x+12,984,x+111,1084),8)
        c.text(x+126,991,f'{i+1:03}',18,MUTED)
        c.text(x+126,1025,glyphs[i],28,TEXT,True)
        c.text(x+13,1100,'待校对' if i==3 else '已确认',16,AMBER if i==3 else GREEN)
    x=1496;fw=378
    c.section(x,196,'分字与识别')
    c.text(x,239,'分析来源：当前去背景结果',17,MUTED)
    c.text(x,281,'排版方向',18,MUTED);c.field(x+154,272,224,'横排',True)
    c.text(x,333,'阅读顺序',18,MUTED);c.field(x+154,324,224,'左 → 右，逐行',True)
    c.text(x,385,'笔画外留白',18,MUTED);c.field(x+225,376,153,'20 px')
    c.check(x,434,'同时识别文字，保留繁体 / 异体')
    c.button(x,480,fw,'分析当前页 / 重新分字','primary',46)
    c.progress(x,545,fw,100,'分析完成 · 2 个字需要人工校对')
    c.line(x,612,x+fw,612)
    c.section(x,633,'当前单字 · 001')
    c.text(x,682,'确认文字',18,MUTED);c.field(x+154,670,224,'住',active=True,h=48)
    c.text(x,742,'导出序号',18,MUTED);c.field(x+154,732,224,'001')
    c.text(x,793,'文件名预览',17,MUTED)
    c.field(x,823,fw,'住-字帖001-001.png')
    c.button(x,880,fw,'确认并选择下一个','green',44)
    c.text(x,943,'字框可拖动；合并 / 拆分只影响裁切，',16,MUTED)
    c.text(x,968,'不重新去杂，不改变文字墨色。',16,MUTED)
    c.button(x,1022,fw,'导出单字…','primary',48)
    c.text(x,1092,'默认只导出已确认的字，待校对项不漏报。',16,MUTED)
    c.save('02-character-review.png')
    return c


def export_screen():
    c=character_screen()
    shade=Image.new('RGB',c.im.size,'#090d13')
    c.im=Image.blend(c.im,shade,.74);c.d=ImageDraw.Draw(c.im)
    x,y=275,154
    c.rect((x+8,y+12,1645,1080),'#090d13',None,16)
    c.rect((x,y,1635,1068),'#1c2633','#50647d',12)
    c.text(x+32,y+24,'导出单字',28,TEXT,True)
    c.text(x+32,y+69,'字帖001  /  12 个候选字，其中 10 个已确认、2 个待校对',18,MUTED)
    c.text(1590,y+26,'×',27,MUTED)
    c.line(x,y+116,1635,y+116)
    lx=x+32;rx=1010
    c.section(lx,294,'导出内容')
    c.field(lx,337,600,'当前图片 · 已确认的 10 个字',True)
    c.check(lx,393,'包含待校对项（未识别字使用占位名称）',False)
    c.text(lx,446,'保存位置',18,MUTED)
    c.field(lx,479,510,r'D:\…\导出\单字')
    c.button(lx+522,479,78,'选择',h=40)
    c.text(lx,545,'文件格式',18,MUTED)
    for i,label in enumerate(('PNG','TIF','PSD','JPG')):
        c.button(lx+i*155,578,135,label,'active' if i==0 else 'normal')
    c.check(lx,638,'添加最底部白底图层（仅 TIF / PSD）',False,True)
    c.text(lx,682,'PNG：透明背景，无损保留墨色。',17,GREEN)
    c.line(lx,725,lx+600,725)
    c.text(lx,747,'裁切与尺寸',20,TEXT,True)
    c.field(lx,791,600,'按字框裁切 + 20 px 留白',True)
    c.check(lx,848,'统一输出分辨率',False)
    c.field(lx+407,839,193,'300 dpi')
    c.text(lx,889,'未勾选：保留当前结果的像素比例与 600 dpi。',16,MUTED)
    c.line(965,294,965,945)
    c.section(rx,294,'命名与导出预览')
    c.text(rx,338,'命名规则',17,MUTED)
    c.field(rx,371,590,'{文字}-{原文件名}-{三位序号}')
    c.text(rx,427,'无法识别：未识别-字帖001-004.png',17,AMBER)
    c.rect((rx,474,1603,756),FIELD,LINE,7)
    c.text(rx+18,491,'将生成的文件',18,TEXT,True)
    filenames=['住-字帖001-001.png','法-字帖001-002.png','所-字帖001-003.png','法-字帖001-005.png','而-字帖001-006.png']
    for i,name in enumerate(filenames):
        yy=536+i*40
        c.line(rx+18,yy+10,rx+23,yy+15,GREEN,2)
        c.line(rx+23,yy+15,rx+31,yy+4,GREEN,2)
        c.text(rx+48,yy,name,18)
    c.text(rx,781,'原序号保留，跳过待校对项不会重新编号。',16,MUTED)
    c.text(rx,832,'目标文件已存在时',17,MUTED)
    c.field(rx,865,590,'询问是否覆盖',True)
    c.text(rx,921,'导出进度按单字显示；支持取消，已完成文件保留。',16,MUTED)
    c.line(x,970,1635,970)
    c.text(lx,1007,'共导出 10 个文件  ·  跳过 2 个待校对项',19,TEXT)
    c.button(1265,993,120,'取消',h=46)
    c.button(1401,993,202,'开始导出 10 个字','primary',46)
    c.save('03-character-export.png')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    page_screen();character_screen();export_screen()
    print(OUT)
