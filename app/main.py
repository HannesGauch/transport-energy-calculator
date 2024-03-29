#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:33:10 2021

@author: Hannes
"""
import numpy as np
import matplotlib.pyplot as plt
from bokeh.models import ColumnDataSource
from bokeh.io import output_file
from bokeh.plotting import gridplot,figure, show
from bokeh.io import curdoc
from bokeh.layouts import column, row,layout,Spacer
from bokeh.models import Slider,ColumnDataSource,Div,Label
from bokeh.core.properties import value
from palettable.colorbrewer.qualitative import Set3_10 as colorpalette



def get_contour_data(X, Y, Z,levels):
    cs = plt.contour(X, Y, Z,levels)
    xs = []
    ys = []
    xt = []
    yt = []
    col = []
    text = []
    isolevelid = 0
    for isolevel in cs.collections:
        isocol = isolevel.get_edgecolor()[0]
        thecol = 3 * [None]
        theiso = str(int(cs.get_array()[isolevelid]))+' TWh'
        isolevelid += 1
        for i in range(3):
            thecol[i] = int(255 * isocol[i])
        thecol = '#%02x%02x%02x' % (thecol[0], thecol[1], thecol[2])

        for path in isolevel.get_paths():
            v = path.vertices
            x = v[:, 0]
            y = v[:, 1]
            xs.append(x.tolist())
            ys.append(y.tolist())
            xt.append(x[int(len(x) / 2)])
            yt.append(y[int(len(y) / 2)])
            text.append(theiso)
            col.append(thecol)

    source = ColumnDataSource(data={'xs': xs, 'ys': ys, 'line_color': col,'xt':xt,'yt':yt,'text':text})
    return source


# output_file("contour.html")
levels = [10,30,50,100,150,200,250]

x = np.arange(0, 1.5, 0.01)
y = np.arange(0, 1000, 1)
X, Y = np.meshgrid(x, y)
Z = X*Y

intensities = {'car electric':0.12,'car hydrogen':0.36,'van electric':0.267,'motorcycle':0.04,'bus electric':0.14,'rail electric':0.08,'rail hydrogen':0.192,'air synthetic':0.9,'air hydrogen':1.4,'bicycle':0.0,'walk':0.0,'HGV electric':0.161,'HGV hydrogen':0.4025,'rail freight electric':0.09}
demands = {'car electric':539,'car hydrogen':0,'van electric':134,'motorcycle':5,'bus electric':29,'rail electric':81,'rail hydrogen':0,'walk':12,'bicycle':6,'air synthetic':0,'HGV electric':161,'HGV hydrogen':0,'rail freight electric':17}
# N = 400
# x = np.linspace(-1, 1, N)
# y = np.linspace(-1, 1, N)
# X, Y = np.meshgrid(x, y)
# Z = X**2 + Y**2


car_dict = {'intensity':[intensities['car electric']],'demand':[demands['car electric']]}
car_source = ColumnDataSource(data=car_dict)
car_source.data['url']=['app/static/electric_car.png']
car_source_fixed = ColumnDataSource()
car_source_fixed.data.update(car_source.data)

carH2_dict = {'intensity':[intensities['car hydrogen']],'demand':[demands['car hydrogen']]}
carH2_source = ColumnDataSource(data=carH2_dict)
carH2_source.data['url']=['app/static/hydrogen_car.png']

bus_dict = {'intensity':[intensities['bus electric']],'demand':[demands['bus electric']]}
bus_source = ColumnDataSource(data=bus_dict)
bus_source.data['url']=['app/static/bus-electric.png']
bus_source_fixed = ColumnDataSource()
bus_source_fixed.data.update(bus_source.data)

plane_dict = {'intensity':[intensities['air synthetic']],'demand':[demands['air synthetic']]}
plane_source = ColumnDataSource(data=plane_dict)
plane_source.data['url']=['app/static/plane-synthetic.png']

train_dict = {'intensity':[intensities['rail electric']],'demand':[demands['rail electric']]}
train_source = ColumnDataSource(data=train_dict)
train_source.data['url']=['app/static/train-electric.png']
train_source_fixed = ColumnDataSource()
train_source_fixed.data.update(train_source.data)

trainH2_dict = {'intensity':[intensities['rail hydrogen']],'demand':[demands['rail hydrogen']]}
trainH2_source = ColumnDataSource(data=trainH2_dict)
trainH2_source.data['url']=['app/static/train-hydrogen.png']

train_freight_dict = {'intensity':[intensities['rail freight electric']],'demand':[demands['rail freight electric']]}
train_freight_source = ColumnDataSource(data=train_freight_dict)
train_freight_source.data['url']=['app/static/train-freight-electric.png']
train_freight_source_fixed = ColumnDataSource()
train_freight_source_fixed.data.update(train_freight_source.data)

walk_dict = {'intensity':[intensities['walk']],'demand':[demands['walk']]}
walk_source = ColumnDataSource(data=walk_dict)
walk_source.data['url']=['app/static/walk-walking.png']
walk_source_fixed = ColumnDataSource()
walk_source_fixed.data.update(walk_source.data)

bike_dict = {'intensity':[intensities['bicycle']],'demand':[demands['bicycle']]}
bike_source = ColumnDataSource(data=bike_dict)
bike_source.data['url']=['app/static/bicycle.png']
bike_source_fixed = ColumnDataSource()
bike_source_fixed.data.update(bike_source.data)

motorcycle_dict = {'intensity':[intensities['motorcycle']],'demand':[demands['motorcycle']]}
motorcycle_source = ColumnDataSource(data=motorcycle_dict)
motorcycle_source.data['url']=['app/static/scooter_electric.png']
motorcycle_source_fixed = ColumnDataSource()
motorcycle_source_fixed.data.update(motorcycle_source.data)

hgvE_dict = {'intensity':[intensities['HGV electric']],'demand':[demands['HGV electric']]}
hgvE_source = ColumnDataSource(data=hgvE_dict)
hgvE_source.data['url']=['app/static/delivery-van-electric.png']
hgvE_source_fixed = ColumnDataSource()
hgvE_source_fixed.data.update(hgvE_source.data)

hgvH2_dict = {'intensity':[intensities['HGV hydrogen']],'demand':[demands['HGV hydrogen']]}
hgvH2_source = ColumnDataSource(data=hgvH2_dict)
hgvH2_source.data['url']=['app/static/delivery-van-hydrogen.png']

van_dict = {'intensity':[intensities['van electric']],'demand':[demands['van electric']]}
van_source = ColumnDataSource(data=van_dict)
van_source.data['url']=['app/static/van_electric.png']
van_source_fixed = ColumnDataSource()
van_source_fixed.data.update(van_source.data)

# contour plot
contour_source = get_contour_data(X,Y,Z,levels)
plot = figure(plot_width=729,plot_height=450,x_range=[-0.04,0.5], y_range=[-20,700],
              x_axis_label='Electric energy intensity [kWh/pkm or kWh/tkm]',y_axis_label='Demand [Gpkm or Gtkm]')
plot.yaxis.axis_label_text_font_style = "normal"
plot.xaxis.axis_label_text_font_style = "normal"
plot.xaxis.axis_label_text_font_size = "12pt"
plot.yaxis.axis_label_text_font_size = "12pt"
plot.multi_line(xs='xs', ys='ys', line_color='line_color', source=contour_source)
plot.text(x='xt',y='yt',text='text',source=contour_source,text_baseline='middle',text_align='center',angle=-0.5,text_font_size='10px')
#plot.circle(intensities['car electric'], demands['car electric'], line_color="yellow", size=12)
#plot.circle(intensities['rail electric'], demands['rail electric'], line_color="yellow", size=12)
#plot.circle(intensities['walk'], demands['walk'], line_color="yellow", size=12)
#plot.circle(intensities['bicycle'], demands['bicycle'], line_color="yellow", size=12)
#plot.circle(intensities['air synthetic'], demands['air synthetic'], line_color="yellow", size=12)

image_widths = {'car':0.07,'bike':0.07,'walk':0.07,'plane':0.07,'train':0.05,'hgv':0.05,'train_freight':0.06,'bus':0.05,'van':0.05,'motorcycle':0.05}
icon_car = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['car'],anchor='center',source=car_source)
icon_carH2 = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['car'],anchor='center',source=carH2_source)

icon_bike = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['bike'],anchor='center',source=bike_source)
icon_walk = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['walk'],anchor='center',source=walk_source)
icon_plane = plot.image_url(url='url',x='intensity',y='demand',h=50,w=image_widths['plane'],anchor='center',source=plane_source)
icon_train = plot.image_url(url='url',x='intensity',y='demand',h=80,w=image_widths['train'],anchor='center',source=train_source)
icon_trainH2 = plot.image_url(url='url',x='intensity',y='demand',h=70,w=image_widths['train'],anchor='center',source=trainH2_source)
icon_hgv = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['hgv'],anchor='center',source=hgvE_source)
icon_hgvH2 = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['hgv'],anchor='center',source=hgvH2_source)
icon_train_freight = plot.image_url(url='url',x='intensity',y='demand',h=50,w=image_widths['train_freight'],anchor='center',source=train_freight_source)
icon_bus = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['bus'],anchor='center',source=bus_source)
icon_van = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['van'],anchor='center',source=van_source)
icon_motorcycle = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['motorcycle'],anchor='center',source=motorcycle_source)

#shadow icons
icon_car0 = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['car'],anchor='center',source=car_source_fixed,alpha=0.3)
icon_bike0 = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['bike'],anchor='center',source=bike_source_fixed,alpha=0.3)
icon_walk0 = plot.image_url(url='url',x='intensity',y='demand',h=40,w=image_widths['walk'],anchor='center',source=walk_source_fixed,alpha=0.3)
icon_train0 = plot.image_url(url='url',x='intensity',y='demand',h=80,w=image_widths['train'],anchor='center',source=train_source_fixed,alpha=0.3)
icon_bus0 = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['bus'],anchor='center',source=bus_source_fixed,alpha=0.3)
icon_van0 = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['van'],anchor='center',source=van_source_fixed,alpha=0.3)
icon_hgv0 = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['hgv'],anchor='center',source=hgvE_source_fixed,alpha=0.3)
icon_train_freight0 = plot.image_url(url='url',x='intensity',y='demand',h=50,w=image_widths['train_freight'],anchor='center',source=train_freight_source_fixed,alpha=0.3)
icon_motorcycle0 = plot.image_url(url='url',x='intensity',y='demand',h=60,w=image_widths['motorcycle'],anchor='center',source=motorcycle_source_fixed,alpha=0.3)

# bar chart
#fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
# names = [0.5]
# colors = ["#c9d9d3", "#718dbf", "#e84d60"]

# heights = [150]

# bar_source = ColumnDataSource(data=dict(names=names, heights=heights))


# barchart = figure(x_range=(0,1), y_range=(0,200), plot_width=200,plot_height=450,
#            toolbar_location=None, tools="",y_axis_label='electricity demand [TWh]')
# barchart.yaxis.axis_label_text_font_style = "normal"

# barchart.vbar(x='names', top='heights', width=0.9, source=bar_source)
# barchart.line([0,1],[90,90],color='chartreuse',line_width=3,line_dash='dashed')
# barchart.xaxis.major_tick_line_color = None
# barchart.xaxis.minor_tick_line_color = None
# barchart.xaxis.major_label_text_font_size = '0pt' 
# barchart.xaxis.major_label_text_font_size = '0pt' 
# label = Label(x=0.5,y=90,text='supply in 2050',text_align='center',text_color='chartreuse')
# barchart.add_layout(label)


# barchart.xgrid.grid_line_color = None


colors = [colorpalette.hex_colors[0],colorpalette.hex_colors[1],colorpalette.hex_colors[2],
          colorpalette.hex_colors[9],colorpalette.hex_colors[3],colorpalette.hex_colors[4],colorpalette.hex_colors[5],
          colorpalette.hex_colors[6],colorpalette.hex_colors[8]]


# stacked
names = [0.5]
modes = ['walk','bicycle','car','van','motorcycle','bus','rail','air','HGV']
#colors = ["#c9d9d3", "#718dbf", "#e84d60"]

data = { 'names' : names,
        'walk'  : [0],
        'bicycle': [0],
        'car'   : [0],
        'van'   : [0],
        'motorcycle':[0],
        'bus'   : [0],
        'rail'   : [0],
        'air'   : [0],
        'HGV'   : [0]}

bar_source = ColumnDataSource(data=data)

barchart = figure(x_range=(0,1), y_range=(0,300), plot_width=300,plot_height=450,
            toolbar_location=None, tools="",y_axis_label='Electricity demand [TWh]')
barchart.yaxis.axis_label_text_font_style = "normal"
barchart.yaxis.axis_label_text_font_size = "12pt"



barchart.vbar_stack(modes, x='names', width=0.9, source=bar_source,color=colors,
             legend=[value(x) for x in modes])

barchart.line([0,1],[90,90],color='black',line_width=3,line_dash='dashed')
barchart.xaxis.major_tick_line_color = None
barchart.xaxis.minor_tick_line_color = None
barchart.xgrid.grid_line_color = None
barchart.xaxis.major_label_text_font_size = '0pt' 
barchart.xaxis.major_label_text_font_size = '0pt' 
label = Label(x=0.5,y=90,text='supply in 2050',text_align='center',text_color='black')
barchart.add_layout(label)
barchart.add_layout(barchart.legend[0], 'right')



bar2_source = ColumnDataSource(data=data)

barchart2 = figure(x_range=(0,1), y_range=(0,200), plot_width=150,plot_height=450,
           toolbar_location=None, tools="",y_axis_label='Passenger transport demand [Gpkm]')
barchart2.yaxis.axis_label_text_font_style = "normal"
barchart2.yaxis.axis_label_text_font_size = "12pt"

barchart2.vbar_stack(modes, x='names', width=0.9, source=bar2_source,color=colors)

barchart2.line([0,1],[808,808],color='black',line_width=3,line_dash='dashed')
barchart2.xaxis.major_tick_line_color = None
barchart2.xaxis.minor_tick_line_color = None
barchart2.xgrid.grid_line_color = None
barchart2.xaxis.major_label_text_font_size = '0pt' 
barchart2.xaxis.major_label_text_font_size = '0pt' 
label = Label(x=0.5,y=808,text='today',text_align='center',text_color='black')
barchart2.add_layout(label)
# barchart2.add_layout(barchart.legend[0], 'right')




# heights2 = [801]

# bar2_source = ColumnDataSource(data=dict(names=names, heights=heights2))


# barchart2 = figure(x_range=(0,1), y_range=(0,200), plot_width=150,plot_height=450,
#            toolbar_location=None, tools="",y_axis_label='passenger transport demand [Gpkm]')
# barchart2.yaxis.axis_label_text_font_style = "normal"

# barchart2.vbar(x='names', top='heights', width=0.9, source=bar2_source)
# barchart2.line([0,1],[801,801],color='chartreuse',line_width=3,line_dash='dashed')
# barchart2.xaxis.major_tick_line_color = None
# barchart2.xaxis.minor_tick_line_color = None
# barchart2.xaxis.major_label_text_font_size = '0pt' 
# barchart2.xaxis.major_label_text_font_size = '0pt' 
# label = Label(x=0.5,y=801,text='today',text_align='center',text_color='chartreuse')
# barchart2.add_layout(label)


# barchart2.xgrid.grid_line_color = None


bar3_source = ColumnDataSource(data=data)

barchart3 = figure(x_range=(0,1), y_range=(0,200), plot_width=150,plot_height=450,
            toolbar_location=None, tools="",y_axis_label='Freight transport demand [Gtkm]')
barchart3.yaxis.axis_label_text_font_style = "normal"
barchart3.yaxis.axis_label_text_font_size = "12pt"

barchart3.vbar_stack(modes, x='names', width=0.9, source=bar3_source,color=colors)

barchart3.line([0,1],[178,178],color='black',line_width=3,line_dash='dashed')
barchart3.xaxis.major_tick_line_color = None
barchart3.xaxis.minor_tick_line_color = None
barchart3.xgrid.grid_line_color = None
barchart3.xaxis.major_label_text_font_size = '0pt' 
barchart3.xaxis.major_label_text_font_size = '0pt' 
label = Label(x=0.5,y=178,text='today',text_align='center',text_color='black')
barchart3.add_layout(label)

# heights3 = [178]

# bar3_source = ColumnDataSource(data=dict(names=names, heights=heights3))


# barchart3 = figure(x_range=(0,1), y_range=(0,200), plot_width=150,plot_height=450,
#            toolbar_location=None, tools="",y_axis_label='freight transport demand [Gtkm]')
# barchart3.yaxis.axis_label_text_font_style = "normal"

# barchart3.vbar(x='names', top='heights', width=0.9, source=bar3_source)
# barchart3.line([0,1],[178,178],color='chartreuse',line_width=3,line_dash='dashed')
# barchart3.xaxis.major_tick_line_color = None
# barchart3.xaxis.minor_tick_line_color = None
# barchart3.xaxis.major_label_text_font_size = '0pt' 
# barchart3.xaxis.major_label_text_font_size = '0pt' 
# label = Label(x=0.5,y=178,text='today',text_align='center',text_color='chartreuse')
# barchart3.add_layout(label)


# barchart3.xgrid.grid_line_color = None

#barchart.legend.orientation = "horizontal"
#barchart.legend.location = "top_center"

div1 = Div(
    text="""
          <p><b>Set demand for private passenger transport [Gpkm]:</b></p>
          """,
    width=200,
    height=50,
    #style={'font-size':'110%'}
)

div1_1 = Div(
    text="""
          <p><b>Set demand for public passenger transport [Gpkm]:</b></p>
          """,
    width=200,
    height=50,
)

div2 = Div(
    text="""
          <p><b>Set demand for freight transport [Gtkm]:</b></p>
          """,
    width=200,
    height=50,
)

div3 = Div(
    text="""
          <p><b>Set average vehicle utilisation:</b></p>
          """,
    width=200,
    height=50,
)

div4 = Div(
    text="""
          <p><b>Set efficiency improvements:</b></p>
          """,
    width=200,
    height=50,
)

div5 = Div(
    text="""
          <p><b>Input levers:</b></p>
          """,
    width=200,
    height=5,
    style={'font-size':'120%'}
)

div5 = Div(
    text="""
          <p><b>Explanations:</b> <br>
          - UK electricity supply for ground transport in 2050 is taken from the <a href="https://ukfires.org/absolute-zero/">UK FIRES Absolute Zero report</a>  <br>
          - The default input numbers (numbers in parantheses and shadow icons) correspond to UK domestic ground transport in 2018 under the assumption that everything is electrified. <br>
          - The energy intensities represent fleet averages <br>
          - The demand numbers were sourced from <a href="https://www.gov.uk/transport#research_and_statistics">UK national statistics </a> <br>
          - Refresh the page to get back to default values <br>
          - For comments and suggestions please contact <a href="mailto:hlg46@cam.ac.uk">hlg46@cam.ac.uk</a>
          </p>
          """,
    width=1100,
    height=20,
    style={'font-size':'100%'}
)

walk_demand = Slider(title="walk ("+str(demands['walk'])+")", value=demands['walk'], start=0.0, end=700.0, step=1,width=200)
bike_demand = Slider(title="bicycle ("+str(demands['bicycle'])+")", value=demands['bicycle'], start=0.0, end=700.0, step=1,width=200)
ecar_demand = Slider(title="car (electric) ("+str(demands['car electric'])+")", value=demands['car electric'], start=0.0, end=700.0, step=1,width=200)
h2car_demand = Slider(title="car (hydrogen) ("+str(demands['car hydrogen'])+")", value=demands['car hydrogen'], start=0.0, end=700.0, step=1,width=200)
evan_demand = Slider(title="van (electric) ("+str(demands['van electric'])+")", value=demands['van electric'], start=0.0, end=700.0, step=1,width=200)
syn_air_demand = Slider(title="air (synthetic fuel) ("+str(demands['air synthetic'])+")", value=demands['air synthetic'], start=0.0, end=1000.0, step=1,width=200)
etrain_demand = Slider(title="rail (electric) ("+str(demands['rail electric'])+")", value=demands['rail electric'], start=0.0, end=700.0, step=1,width=200)
h2train_demand = Slider(title="rail (hydrogen) ("+str(demands['rail hydrogen'])+")", value=demands['rail hydrogen'], start=0.0, end=700.0, step=1,width=200)
etrain_freight_demand = Slider(title="rail (electric) ("+str(demands['rail freight electric'])+")", value=demands['rail freight electric'], start=0.0, end=300.0, step=1,width=200)
bus_demand = Slider(title="bus (electric) ("+str(demands['bus electric'])+")", value=demands['bus electric'], start=0.0, end=700.0, step=1,width=200)
motorcycle_demand = Slider(title="motorcycle (electric) ("+str(demands['motorcycle'])+")", value=demands['motorcycle'], start=0.0, end=700.0, step=1,width=200)

eHGV_demand = Slider(title="HGV (electric) ("+str(demands['HGV electric'])+")", value=demands['HGV electric'], start=0.0, end=300.0, step=1,width=200)
h2HGV_demand = Slider(title="HGV (hydrogen) ("+str(demands['HGV hydrogen'])+")", value=demands['HGV hydrogen'], start=0.0, end=300.0, step=1,width=200)


car_util = Slider(title="passengers per car (1.5)", value=1.5, start=1, end=4, step=0.1,width=200)
train_util = Slider(title="train occupancy rate [%] (50)", value=50, start=1, end=100, step=1,width=200)
bus_util = Slider(title="bus occupancy rate [%] (50)", value=50, start=1, end=100, step=1,width=200)
HGV_util = Slider(title="HGV average load factor (0.5)", value=0.5, start=0.1, end=1, step=0.05,width=200)
train_freight_util = Slider(title="freight train load factor (0.5)", value=0.5, start=0.1, end=1, step=0.05,width=200)


car_weight = Slider(title="average car weight [kg] (1400)", value=1400, start=700, end=2100, step=50,width=200)
#car_RB = Slider(title="cars: exploiting reg. breaking [%]", value=0, start=0, end=100, step=5,width=200)
#car_DF = Slider(title="cars: reduced drag & friction [%]", value=0, start=0, end=100, step=5,width=200)
#train_RB = Slider(title="trains: exploiting reg. breaking [%]", value=0, start=0, end=100, step=5,width=200)
#train_DF = Slider(title="trains: reduced drag & friction [%]", value=0, start=0, end=100, step=5,width=200)
#HGV_RB = Slider(title="HGV/LGV: exploiting reg. breaking [%]", value=0, start=0, end=100, step=5,width=200)
#HGV_DF = Slider(title="HGV/LGV: reduced drag & friction [%]", value=0, start=0, end=100, step=5,width=200)
reg_break = Slider(title="regenerative breaking [% expl.]", value=0, start=0, end=100, step=5,width=200)
drag_fric = Slider(title="reduced drag & friction [% expl.]", value=0, start=0, end=100, step=5,width=200)

inputs = row(column(div1,walk_demand,bike_demand,ecar_demand,h2car_demand,evan_demand,motorcycle_demand),
              column(div1_1,bus_demand,etrain_demand,h2train_demand,syn_air_demand),
              column(div2,etrain_freight_demand,eHGV_demand,h2HGV_demand),
              column(div3,car_util,train_util,bus_util,HGV_util,train_freight_util),
              column(div4,car_weight,reg_break,drag_fric))


def update_data(changed):
    
    RB_factor = (0.85-1)/(100)*(reg_break.value)+1
    DF_factor = (0.95-1)/(100)*(drag_fric.value)+1

    if 'car' in changed:
        car_util_factor = 1.5 / car_util.value
        car_weight_factor = (0.85-1)/(1000-1400)*(car_weight.value-1400)+1    
    
        ecar_d = ecar_demand.value
        car_source.data['demand'] = [ecar_d]
        car_source.data['intensity'] = [intensities['car electric'] * car_util_factor * car_weight_factor*RB_factor*DF_factor]
        
        h2car_d = h2car_demand.value
        carH2_source.data['demand'] = [h2car_d]
        carH2_source.data['intensity'] = [intensities['car hydrogen'] * car_util_factor * car_weight_factor*RB_factor*DF_factor]
        
        bar_source.data['car'] = [car_source.data['demand'][0]*car_source.data['intensity'][0] + carH2_source.data['demand'][0]*carH2_source.data['intensity'][0]]
        bar2_source.data['car'] = [car_source.data['demand'][0] + carH2_source.data['demand'][0]]
        
        if(h2car_demand.value<1):
            carH2_source.data['url']=['']
        else:
            carH2_source.data['url']=['app/static/hydrogen_car.png']

        
    if 'bus' in changed:

        bus_source.data['demand'] = [bus_demand.value]
        bus_source.data['intensity'] = [intensities['bus electric'] * 0.5 / (bus_util.value/100)*RB_factor*DF_factor]
        bar_source.data['bus'] = [bus_source.data['demand'][0]*bus_source.data['intensity'][0]]
        bar2_source.data['bus'] = [bus_source.data['demand'][0]]



    if 'van' in changed:
        van_d = evan_demand.value
        van_source.data['demand'] = [van_d]
        van_source.data['intensity'] = [intensities['van electric'] *RB_factor*DF_factor]
        bar_source.data['van'] = [van_source.data['demand'][0]*van_source.data['intensity'][0]]
        bar2_source.data['van'] = [van_source.data['demand'][0]]

        

     
    if 'air' in changed:

        synair_d = syn_air_demand.value
        plane_source.data['demand'] = [synair_d]
        bar_source.data['air'] = [plane_source.data['demand'][0]*plane_source.data['intensity'][0]]
        
        bar2_source.data['air'] = [plane_source.data['demand'][0]]
        
        if(syn_air_demand.value<1):
            plane_source.data['url']=['']
        else:
            plane_source.data['url']=['app/static/plane-synthetic.png']

        
        if plane_source.data['demand'][0] > 1e-8:
            plot.x_range.end = 1.0
            icon_car.glyph.w = image_widths['car']
            icon_carH2.glyph.w = image_widths['car']   
            icon_bike.glyph.w = image_widths['bike']
            icon_walk.glyph.w = image_widths['walk']
            icon_plane.glyph.w = image_widths['plane']
            icon_train.glyph.w = image_widths['train']
            icon_trainH2.glyph.w = image_widths['train']
            icon_hgv.glyph.w = image_widths['hgv']
            icon_hgvH2.glyph.w = image_widths['hgv']
            icon_train_freight.glyph.w = image_widths['train_freight']
            icon_bus.glyph.w = image_widths['bus']
            icon_van.glyph.w = image_widths['van']
            icon_motorcycle.glyph.w = image_widths['motorcycle']
            
            #shadow icons
            icon_car0.glyph.w = image_widths['car']
            icon_bike0.glyph.w = image_widths['bike']
            icon_walk0.glyph.w = image_widths['walk']
            icon_train0.glyph.w = image_widths['train']
            icon_bus0.glyph.w = image_widths['bus']
            icon_van0.glyph.w = image_widths['van']
            icon_hgv0.glyph.w = image_widths['hgv']
            icon_train_freight0.glyph.w = image_widths['train_freight']
            icon_motorcycle0.glyph.w = image_widths['motorcycle']

        else:
            plot.x_range.end = 0.5
            icon_car.glyph.w = image_widths['car']/2.
            icon_carH2.glyph.w = image_widths['car']/2.     
            icon_bike.glyph.w = image_widths['bike']/2.
            icon_walk.glyph.w = image_widths['walk']/2.
            icon_plane.glyph.w = image_widths['plane']/2.
            icon_train.glyph.w = image_widths['train']/2.
            icon_trainH2.glyph.w = image_widths['train']/2.
            icon_hgv.glyph.w = image_widths['hgv']/2.
            icon_hgvH2.glyph.w = image_widths['hgv']/2.
            icon_train_freight.glyph.w = image_widths['train_freight']/2.
            icon_bus.glyph.w = image_widths['bus']/2.
            icon_van.glyph.w = image_widths['van']/2.
            icon_motorcycle.glyph.w = image_widths['motorcycle']/2.
            
            #shadow icons
            icon_car0.glyph.w = image_widths['car']/2.
            icon_bike0.glyph.w = image_widths['bike']/2.
            icon_walk0.glyph.w = image_widths['walk']/2.
            icon_train0.glyph.w = image_widths['train']/2.
            icon_bus0.glyph.w = image_widths['bus']/2.
            icon_van0.glyph.w = image_widths['van']/2.
            icon_hgv0.glyph.w = image_widths['hgv']/2.
            icon_train_freight0.glyph.w = image_widths['train_freight']/2.
            icon_motorcycle0.glyph.w = image_widths['motorcycle']/2.

        
    if 'train' in changed:
    
        etrain_d = etrain_demand.value
        train_source.data['demand'] = [etrain_d]
        train_source.data['intensity'] = [intensities['rail electric'] * 0.5 / (train_util.value/100)*RB_factor*DF_factor]
        
        trainH2_source.data['demand'] = [h2train_demand.value]
        trainH2_source.data['intensity'] = [intensities['rail hydrogen'] * 0.5 / (train_util.value/100)*RB_factor*DF_factor]
    
        
        train_freight_source.data['demand'] = [etrain_freight_demand.value]
        train_freight_source.data['intensity'] = [intensities['rail freight electric'] * 0.5 / (train_freight_util.value)*RB_factor*DF_factor]
        
        bar_source.data['rail'] = [train_source.data['demand'][0]*train_source.data['intensity'][0]
                                   +train_freight_source.data['demand'][0]*train_freight_source.data['intensity'][0]
                                   +trainH2_source.data['demand'][0]*trainH2_source.data['intensity'][0]]
        
        bar2_source.data['rail'] = [train_source.data['demand'][0]            
                                    + trainH2_source.data['demand'][0]]
        
        bar3_source.data['rail'] = [train_freight_source.data['demand'][0]]
        
        if(h2train_demand.value<1):
            trainH2_source.data['url']=['']
        else:
            trainH2_source.data['url']=['app/static/train-hydrogen.png']


        
    if 'hgv' in changed:
        hgvE_source.data['demand'] = [eHGV_demand.value]
        hgvE_source.data['intensity'] = [intensities['HGV electric'] * 0.5 / (HGV_util.value)*RB_factor*DF_factor]
        
        hgvH2_source.data['demand'] = [h2HGV_demand.value]
        hgvH2_source.data['intensity'] = [intensities['HGV hydrogen'] * 0.5 / (HGV_util.value)*RB_factor*DF_factor]
        
        bar_source.data['HGV'] = [hgvE_source.data['demand'][0]*hgvE_source.data['intensity'][0]
                                  + hgvH2_source.data['demand'][0]*hgvH2_source.data['intensity'][0]]
        bar3_source.data['HGV'] = [hgvE_source.data['demand'][0]+hgvH2_source.data['demand'][0]]
        
        if(h2HGV_demand.value<1):
            hgvH2_source.data['url']=['']
        else:
            hgvH2_source.data['url']=['app/static/delivery-van-hydrogen.png']

    
    if 'walk/bike' in changed:
    
        walk_d = walk_demand.value
        walk_source.data['demand'] = [walk_d]
        
        bike_d = bike_demand.value
        bike_source.data['demand'] = [bike_d]
        
        bar2_source.data['walk'] = [walk_demand.value]
        bar2_source.data['bicycle'] = [bike_demand.value]
    
    if 'motorcycle' in changed:
        motorcycle_d = motorcycle_demand.value
        motorcycle_source.data['demand'] = [motorcycle_d]
        bar_source.data['motorcycle'] = [motorcycle_source.data['demand'][0]*motorcycle_source.data['intensity'][0]]
        bar2_source.data['motorcycle'] = [motorcycle_source.data['demand'][0]]


    
    Etot = (car_source.data['demand'][0]*car_source.data['intensity'][0]
            + carH2_source.data['demand'][0]*carH2_source.data['intensity'][0]
            + plane_source.data['demand'][0]*plane_source.data['intensity'][0]
            + train_source.data['demand'][0]*train_source.data['intensity'][0]
            + hgvE_source.data['demand'][0]*hgvE_source.data['intensity'][0]
            + hgvH2_source.data['demand'][0]*hgvH2_source.data['intensity'][0]
            + train_freight_source.data['demand'][0]*train_freight_source.data['intensity'][0]
            + trainH2_source.data['demand'][0]*trainH2_source.data['intensity'][0]
            + bus_source.data['demand'][0]*bus_source.data['intensity'][0]
            + van_source.data['demand'][0]*van_source.data['intensity'][0]
            + motorcycle_source.data['demand'][0]*motorcycle_source.data['intensity'][0])
    
    
    Gpkmtot = (car_source.data['demand'][0]
                   + carH2_source.data['demand'][0]
                   + plane_source.data['demand'][0]
                   + train_source.data['demand'][0]
                   + trainH2_source.data['demand'][0]
                   + bus_source.data['demand'][0]
                   + van_source.data['demand'][0]
                   + walk_source.data['demand'][0]
                   + bike_source.data['demand'][0]
                   + motorcycle_source.data['demand'][0])
    
    Gtkmtot = (hgvE_source.data['demand'][0]
                   + hgvH2_source.data['demand'][0]
                   + train_freight_source.data['demand'][0])
    
    # print(heights3[0])


    barchart.y_range.end = max(Etot+20,300)   
    barchart2.y_range.end = max(Gpkmtot+100,1000)    
    barchart3.y_range.end = max(Gtkmtot+40,222)
    
            
    

    #source.data = dict(x=x, y=y)

# for w in [ecar_demand,h2car_demand,evan_demand,bus_demand,bus_util,motorcycle_demand,syn_air_demand,etrain_demand,walk_demand,bike_demand,h2train_demand,car_util,train_util,h2HGV_demand,eHGV_demand,HGV_util,etrain_freight_demand,train_freight_util,car_weight,reg_break,drag_fric]:
#     w.on_change('value', lambda attr, old, new: update_data())

for w in [ecar_demand,h2car_demand,car_util,car_weight]:
    w.on_change('value', lambda attr, old, new: update_data(['car']))
    
for w in [evan_demand]:
    w.on_change('value', lambda attr, old, new: update_data(['van']))
    
for w in [bus_demand,bus_util]:
    w.on_change('value', lambda attr, old, new: update_data(['bus']))
    
for w in [motorcycle_demand]:
    w.on_change('value', lambda attr, old, new: update_data(['motorcycle']))
    
for w in [syn_air_demand]:
    w.on_change('value', lambda attr, old, new: update_data(['air']))

for w in [etrain_demand,h2train_demand,train_util,etrain_freight_demand,train_freight_util]:
    w.on_change('value', lambda attr, old, new: update_data(['train']))
    
for w in [walk_demand,bike_demand]:
    w.on_change('value', lambda attr, old, new: update_data(['walk/bike']))
    
for w in [h2HGV_demand,eHGV_demand,HGV_util]:
    w.on_change('value', lambda attr, old, new: update_data(['hgv']))
    
for w in [reg_break,drag_fric]:
    w.on_change('value', lambda attr, old, new: update_data(['car','van','bus','train','hgv']))


update_data(['car','van','bus','train','hgv','air','walk/bike','motorcycle'])

#curdoc().add_root(column(row(plot,barchart),inputs) )
curdoc().add_root(layout(children=[[plot,Spacer(width=50),barchart,barchart2,barchart3],[inputs],[div5]],sizing_mode='fixed'))
#curdoc().add_root(gridplot([[plot, barchart], [inputs1, inputs2]], sizing_mode='scale_both'))


#show(row(plot,barchart),sizing_mode='scale_width')