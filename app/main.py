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
        isocol = isolevel.get_color()[0]
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

intensities = {'car electric':0.15,'car hydrogen':0.36,'van electric':0.267,'bus electric':0.14,'rail electric':0.08,'rail hydrogen':0.192,'air synthetic':0.9,'air hydrogen':1.4,'bicycle':0.0,'walk':0.0,'HGV electric':0.361,'HGV hydrogen':0.889,'rail freight electric':0.09}
demands = {'car electric':539,'car hydrogen':0,'van electric':134,'bus electric':29,'rail electric':81,'rail hydrogen':0,'walk':12,'bicycle':6,'air synthetic':0,'HGV electric':161,'HGV hydrogen':0,'rail freight electric':17}
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
plot = figure(plot_width=729,plot_height=450,x_range=[-0.04,1], y_range=[-20,700],
              x_axis_label='energy intensity [kWh/pkm or kWh/tkm]',y_axis_label='demand [Gpkm or Gtkm]')
plot.yaxis.axis_label_text_font_style = "normal"
plot.xaxis.axis_label_text_font_style = "normal"


plot.multi_line(xs='xs', ys='ys', line_color='line_color', source=contour_source)
plot.text(x='xt',y='yt',text='text',source=contour_source,text_baseline='middle',text_align='center',angle=-0.5,text_font_size='10px')
#plot.circle(intensities['car electric'], demands['car electric'], line_color="yellow", size=12)
#plot.circle(intensities['rail electric'], demands['rail electric'], line_color="yellow", size=12)
#plot.circle(intensities['walk'], demands['walk'], line_color="yellow", size=12)
#plot.circle(intensities['bicycle'], demands['bicycle'], line_color="yellow", size=12)
#plot.circle(intensities['air synthetic'], demands['air synthetic'], line_color="yellow", size=12)
plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=car_source)
plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=carH2_source)

plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=bike_source)
plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=walk_source)
plot.image_url(url='url',x='intensity',y='demand',h=50,w=0.07,anchor='center',source=plane_source)
plot.image_url(url='url',x='intensity',y='demand',h=80,w=0.05,anchor='center',source=train_source)
plot.image_url(url='url',x='intensity',y='demand',h=70,w=0.05,anchor='center',source=trainH2_source)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=hgvE_source)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=hgvH2_source)
plot.image_url(url='url',x='intensity',y='demand',h=50,w=0.06,anchor='center',source=train_freight_source)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=bus_source)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=van_source)
#shadow icons
plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=car_source_fixed,alpha=0.3)
plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=bike_source_fixed,alpha=0.3)
plot.image_url(url='url',x='intensity',y='demand',h=40,w=0.07,anchor='center',source=walk_source_fixed,alpha=0.3)
plot.image_url(url='url',x='intensity',y='demand',h=80,w=0.05,anchor='center',source=train_source_fixed,alpha=0.3)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=bus_source_fixed,alpha=0.3)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=van_source_fixed,alpha=0.3)
plot.image_url(url='url',x='intensity',y='demand',h=60,w=0.05,anchor='center',source=hgvE_source_fixed,alpha=0.3)

# bar chart
#fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
names = [0.5]
colors = ["#c9d9d3", "#718dbf", "#e84d60"]

heights = [150]

bar_source = ColumnDataSource(data=dict(names=names, heights=heights))


barchart = figure(x_range=(0,1), y_range=(0,200), plot_width=200,plot_height=450,
           toolbar_location=None, tools="",y_axis_label='electricity demand [TWh]')
barchart.yaxis.axis_label_text_font_style = "normal"

barchart.vbar(x='names', top='heights', width=0.9, source=bar_source)
barchart.line([0,1],[90,90],color='chartreuse',line_width=3,line_dash='dashed')
barchart.xaxis.major_tick_line_color = None
barchart.xaxis.minor_tick_line_color = None
barchart.xaxis.major_label_text_font_size = '0pt' 
barchart.xaxis.major_label_text_font_size = '0pt' 
label = Label(x=0.5,y=90,text='supply in 2050',text_align='center',text_color='chartreuse')
barchart.add_layout(label)


barchart.xgrid.grid_line_color = None



heights2 = [801]

bar2_source = ColumnDataSource(data=dict(names=names, heights=heights2))


barchart2 = figure(x_range=(0,1), y_range=(0,200), plot_width=150,plot_height=450,
           toolbar_location=None, tools="",y_axis_label='passenger transport demand [Gpkm]')
barchart2.yaxis.axis_label_text_font_style = "normal"

barchart2.vbar(x='names', top='heights', width=0.9, source=bar2_source)
barchart2.line([0,1],[801,801],color='chartreuse',line_width=3,line_dash='dashed')
barchart2.xaxis.major_tick_line_color = None
barchart2.xaxis.minor_tick_line_color = None
barchart2.xaxis.major_label_text_font_size = '0pt' 
barchart2.xaxis.major_label_text_font_size = '0pt' 
label = Label(x=0.5,y=801,text='today',text_align='center',text_color='chartreuse')
barchart2.add_layout(label)


barchart2.xgrid.grid_line_color = None


heights3 = [178]

bar3_source = ColumnDataSource(data=dict(names=names, heights=heights3))


barchart3 = figure(x_range=(0,1), y_range=(0,200), plot_width=150,plot_height=450,
           toolbar_location=None, tools="",y_axis_label='freight transport demand [Gtkm]')
barchart3.yaxis.axis_label_text_font_style = "normal"

barchart3.vbar(x='names', top='heights', width=0.9, source=bar3_source)
barchart3.line([0,1],[178,178],color='chartreuse',line_width=3,line_dash='dashed')
barchart3.xaxis.major_tick_line_color = None
barchart3.xaxis.minor_tick_line_color = None
barchart3.xaxis.major_label_text_font_size = '0pt' 
barchart3.xaxis.major_label_text_font_size = '0pt' 
label = Label(x=0.5,y=178,text='today',text_align='center',text_color='chartreuse')
barchart3.add_layout(label)


barchart3.xgrid.grid_line_color = None

#barchart.legend.orientation = "horizontal"
#barchart.legend.location = "top_center"

div1 = Div(
    text="""
          <p>Demand for private passenger transport [Gpkm]:</p>
          """,
    width=200,
    height=20,
)

div1_1 = Div(
    text="""
          <p>Demand for public passenger transport [Gpkm]:</p>
          """,
    width=200,
    height=20,
)

div2 = Div(
    text="""
          <p>Demand for freight transport [Gtkm]:</p>
          """,
    width=200,
    height=20,
)

div3 = Div(
    text="""
          <p>Vehicle utilisation:</p>
          """,
    width=200,
    height=20,
)

div4 = Div(
    text="""
          <p> Efficiency improvements:</p>
          """,
    width=200,
    height=20,
)

walk_demand = Slider(title="walk ("+str(demands['walk'])+")", value=demands['walk'], start=0.0, end=1000.0, step=1,width=200)
bike_demand = Slider(title="bicycle ("+str(demands['bicycle'])+")", value=demands['bicycle'], start=0.0, end=1000.0, step=1,width=200)
ecar_demand = Slider(title="car (electric) ("+str(demands['car electric'])+")", value=demands['car electric'], start=0.0, end=1000.0, step=1,width=200)
h2car_demand = Slider(title="car (hydrogen)("+str(demands['car hydrogen'])+")", value=demands['car hydrogen'], start=0.0, end=1000.0, step=1,width=200)
evan_demand = Slider(title="van (electric)("+str(demands['van electric'])+")", value=demands['van electric'], start=0.0, end=1000.0, step=1,width=200)
syn_air_demand = Slider(title="air (synthetic fuel)("+str(demands['air synthetic'])+")", value=demands['air synthetic'], start=0.0, end=1000.0, step=1,width=200)
etrain_demand = Slider(title="rail (electric)("+str(demands['rail electric'])+")", value=demands['rail electric'], start=0.0, end=1000.0, step=1,width=200)
h2train_demand = Slider(title="rail (hydrogen)("+str(demands['rail hydrogen'])+")", value=demands['rail hydrogen'], start=0.0, end=1000.0, step=1,width=200)
etrain_freight_demand = Slider(title="rail (electric)("+str(demands['rail freight electric'])+")", value=demands['rail freight electric'], start=0.0, end=1000.0, step=1,width=200)
bus_demand = Slider(title="bus (electric)("+str(demands['bus electric'])+")", value=demands['bus electric'], start=0.0, end=1000.0, step=1,width=200)

eHGV_demand = Slider(title="HGV/LGV (electric)("+str(demands['HGV electric'])+")", value=demands['HGV electric'], start=0.0, end=1000.0, step=1,width=200)
h2HGV_demand = Slider(title="HGV/LGV (hydrogen)("+str(demands['HGV hydrogen'])+")", value=demands['HGV hydrogen'], start=0.0, end=1000.0, step=1,width=200)


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

inputs = row(column(div1,walk_demand,bike_demand,ecar_demand,h2car_demand,evan_demand),
              column(div1_1,bus_demand,etrain_demand,h2train_demand,syn_air_demand),
              column(div2,etrain_freight_demand,eHGV_demand,h2HGV_demand),
              column(div3,car_util,train_util,bus_util,HGV_util,train_freight_util),
              column(div4,car_weight,reg_break,drag_fric))


def update_data():

    # Get the current slider values
    car_util_factor = 1.5 / car_util.value
    car_weight_factor = (0.85-1)/(1000-1400)*(car_weight.value-1400)+1
    RB_factor = (0.85-1)/(100)*(reg_break.value)+1
    DF_factor = (0.95-1)/(100)*(drag_fric.value)+1

    
    ecar_d = ecar_demand.value
    car_source.data['demand'] = [ecar_d]
    car_source.data['intensity'] = [intensities['car electric'] * car_util_factor * car_weight_factor*RB_factor*DF_factor]
    
    h2car_d = h2car_demand.value
    carH2_source.data['demand'] = [h2car_d]
    carH2_source.data['intensity'] = [intensities['car hydrogen'] * car_util_factor * car_weight_factor*RB_factor*DF_factor]

    bus_source.data['demand'] = [bus_demand.value]
    bus_source.data['intensity'] = [intensities['bus electric'] * 0.5 / (bus_util.value/100)*RB_factor*DF_factor]

    van_d = evan_demand.value
    van_source.data['demand'] = [van_d]
    van_source.data['intensity'] = [intensities['van electric'] *RB_factor*DF_factor]
     


    synair_d = syn_air_demand.value
    plane_source.data['demand'] = [synair_d]
    
    etrain_d = etrain_demand.value
    train_source.data['demand'] = [etrain_d]
    train_source.data['intensity'] = [intensities['rail electric'] * 0.5 / (train_util.value/100)*RB_factor*DF_factor]
    
    
    hgvE_source.data['demand'] = [eHGV_demand.value]
    hgvE_source.data['intensity'] = [intensities['HGV electric'] * 0.5 / (HGV_util.value)*RB_factor*DF_factor]
    
    hgvH2_source.data['demand'] = [h2HGV_demand.value]
    hgvH2_source.data['intensity'] = [intensities['HGV hydrogen'] * 0.5 / (HGV_util.value)*RB_factor*DF_factor]
    
    etrain_d = etrain_demand.value
    train_source.data['demand'] = [etrain_d]
    train_source.data['intensity'] = [intensities['rail electric'] * 0.5 / (train_util.value/100)*RB_factor*DF_factor]
    
    trainH2_source.data['demand'] = [h2train_demand.value]
    trainH2_source.data['intensity'] = [intensities['rail hydrogen'] * 0.5 / (train_util.value/100)*RB_factor*DF_factor]

    
    train_freight_source.data['demand'] = [etrain_freight_demand.value]
    train_freight_source.data['intensity'] = [intensities['rail freight electric'] * 0.5 / (train_freight_util.value)*RB_factor*DF_factor]

    
    walk_d = walk_demand.value
    walk_source.data['demand'] = [walk_d]
    
    bike_d = bike_demand.value
    bike_source.data['demand'] = [bike_d]
    
    heights[0] = car_source.data['demand'][0]*car_source.data['intensity'][0]
    heights[0] += carH2_source.data['demand'][0]*carH2_source.data['intensity'][0]
    heights[0] += plane_source.data['demand'][0]*plane_source.data['intensity'][0]
    heights[0] += train_source.data['demand'][0]*train_source.data['intensity'][0]
    heights[0] += hgvE_source.data['demand'][0]*hgvE_source.data['intensity'][0]
    heights[0] += hgvH2_source.data['demand'][0]*hgvH2_source.data['intensity'][0]
    heights[0] += train_freight_source.data['demand'][0]*train_freight_source.data['intensity'][0]
    heights[0] += trainH2_source.data['demand'][0]*trainH2_source.data['intensity'][0]
    heights[0] += bus_source.data['demand'][0]*bus_source.data['intensity'][0]
    heights[0] += van_source.data['demand'][0]*van_source.data['intensity'][0]
    
    
    heights2[0] = (car_source.data['demand'][0]
                   + carH2_source.data['demand'][0]
                   + plane_source.data['demand'][0]
                   + train_source.data['demand'][0]
                   + trainH2_source.data['demand'][0]
                   + bus_source.data['demand'][0]
                   + van_source.data['demand'][0]
                   + walk_source.data['demand'][0]
                   + bike_source.data['demand'][0])
    
    heights3[0] = (hgvE_source.data['demand'][0]
                   + hgvH2_source.data['demand'][0]
                   + train_freight_source.data['demand'][0])
    
    # print(heights3[0])


    bar_source.data['heights'] = heights
    barchart.y_range.end = max(heights[0]+20,200)
    
    bar2_source.data['heights'] = heights2
    barchart2.y_range.end = max(heights2[0]+100,1000)
    
    bar3_source.data['heights'] = heights3
    barchart3.y_range.end = max(heights3[0]+40,222)
    
    
    # no icons if demand = 0
    if(syn_air_demand.value<1):
        plane_source.data['url']=['']
    else:
        plane_source.data['url']=['app/static/plane-synthetic.png']
        
    if(h2car_demand.value<1):
        carH2_source.data['url']=['']
    else:
        carH2_source.data['url']=['app/static/hydrogen_car.png']
        
    if(h2HGV_demand.value<1):
        hgvH2_source.data['url']=['']
    else:
        hgvH2_source.data['url']=['app/static/delivery-van-hydrogen.png']
        
    if(h2train_demand.value<1):
        trainH2_source.data['url']=['']
    else:
        trainH2_source.data['url']=['app/static/train-hydrogen.png']

        
    

    #source.data = dict(x=x, y=y)

for w in [ecar_demand,h2car_demand,evan_demand,bus_demand,bus_util,syn_air_demand,etrain_demand,walk_demand,bike_demand,h2train_demand,car_util,train_util,h2HGV_demand,eHGV_demand,HGV_util,etrain_freight_demand,train_freight_util,car_weight,reg_break,drag_fric]:
    w.on_change('value', lambda attr, old, new: update_data())

update_data()

#curdoc().add_root(column(row(plot,barchart),inputs) )
curdoc().add_root(layout(children=[[plot,Spacer(width=50),barchart,barchart2,barchart3],[inputs]],sizing_mode='fixed'))
#curdoc().add_root(gridplot([[plot, barchart], [inputs1, inputs2]], sizing_mode='scale_both'))


#show(row(plot,barchart),sizing_mode='scale_width')