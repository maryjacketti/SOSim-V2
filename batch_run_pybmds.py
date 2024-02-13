import bmds
from bmds import DichotomousDataset
from bmds import ContinuousDataset
from bmds.bmds3.models import dichotomous, continuous
from bmds.bmds3.constants import DistType, PriorClass, PriorType
from bmds.bmds3.models.continuous import ContinuousModel
from bmds.bmds3.types.continuous import ContinuousRiskType
from bmds.bmds3.types.dichotomous import DichotomousRiskType
from bmds.bmds3.types.continuous import ContinuousAnalysis, ContinuousModelSettings, ContinuousResult
from bmds.bmds3.types.priors import ModelPriors, Prior
from itertools import cycle
from bmds import plotting
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline, interp1d


from copy import deepcopy
from datetime import datetime
import itertools
import json
import os
import time
import pandas as pd
import numpy as np

start_time = datetime.now()

print(start_time)

OUTPUT_FN = 'C:/Users/mjacketti/OneDrive - Applied Research Associates, Inc/Desktop/BMDS/HERA3.5.1/Model_Averaging/M2/bmds_M2_result_1.xlsx'

fn = 'C:/Users/mjacketti/OneDrive - Applied Research Associates, Inc/Desktop/BMDS/HERA3.5.1/Model_Averaging/M2/M2.csv'
assert os.path.exists(fn)

df = pd.read_csv(fn)
df = df.iloc[1:2000,:]
df.head()


def dichotomous_dictify(d):
    try:
        return dict(
              id=d.Index,
              doses=list(map(float, d.dose.split(';'))),
              ns=list(map(int, d['N'].split(';'))),
              incidences=list(map(int, d['incidence'].split(';'))),
        )
    except:
        print('Row {} not included'.format(d.Index))
        return None
   
dichotomous_datasets = [
    d for d in df.apply(dichotomous_dictify, axis=1) 
    if d is not None
]

print(len(dichotomous_datasets))

# print(dichotomous_datasets)
# for i in range(len(dichotomous_datasets)):
#     #cubic_interpolation_model = make_interp_spline(dichotomous_datasets[i]['doses'] ,[x/y for x,y in zip(dichotomous_datasets[i]['incidences'],dichotomous_datasets[i]['ns'])])
#     #X_Y_Spline = make_interp_spline(dichotomous_datasets[i]['doses'], [x/y for x,y in zip(dichotomous_datasets[i]['incidences'],dichotomous_datasets[i]['ns'])])
#     #xnew = np.linspace(min(dichotomous_datasets[i]['doses']), max(dichotomous_datasets[i]['doses']), 1000) 
#     #spl = make_interp_spline(dichotomous_datasets[i]['doses'], [x/y for x,y in zip(dichotomous_datasets[i]['incidences'],dichotomous_datasets[i]['ns'])], k=2)  # type: BSpline
#     #power_smooth = spl(xnew)
#     #power_smooth = X_Y_Spline(xnew)
#     #power_smooth = cubic_interpolation_model(xnew)
#     #plt.plot(xnew, power_smooth)
#     plt.scatter(dichotomous_datasets[i]['doses'] ,[x/y for x,y in zip(dichotomous_datasets[i]['incidences'],dichotomous_datasets[i]['ns'])],label = 'id %s'%i)
# plt.savefig('datsets_1plot.png')

def get_bmds_dataset(dtype, dataset):
    if dtype == bmds.constants.CONTINUOUS:
        cls = bmds.ContinuousDataset
    elif dtype in bmds.constants.DICHOTOMOUS_DTYPES:
        cls = bmds.DichotomousDataset
    return cls(**dataset)

def execute_all_datasets(dtype, original_dataset):
    dataset = deepcopy(original_dataset)
    print(dataset['id'])
    bmds_dataset = get_bmds_dataset(dtype, dataset)
    session = bmds.BMDS.latest_version(dataset=bmds_dataset)
    session.add_default_bayesian_models()

    # session = dichotomous.Weibull(dataset=bmds_dataset, settings={"priors": PriorClass.bayesian, "samples": 1000, "burnin": 500})
    # g = session.settings.priors.get_prior('g')
    # g.type = PriorType.Lognormal
    # g.min_value=0
    # g.max_value=1
    # g.initial_value=0
    # g.stdev=1.5
    # b = session.settings.priors.get_prior('b')
    # b.type = PriorType.Uniform
    # b.min_value=0
    # b.max_value=15
    # b.initial_value=5
    # b.stdev = 10
    # a = session.settings.priors.get_prior('a')
    # a.type = PriorType.Uniform
    # a.min_value=-5
    # a.max_value=15
    # a.initial_value=3
    # a.stdev = 10

    #session.add_default_models()
    #session.execute_and_recommend(drop_doses=True)
    result = session.execute()
    res = session.model_average.results
    print(f"BMD = {res.bmd:.2f} [{res.bmdl:.2f}, {res.bmdu:.2f}]")
    return res


dichotomous_results = [
    execute_all_datasets(bmds.constants.DICHOTOMOUS, dataset) 
    for dataset in dichotomous_datasets
]

def get_od():
    # return an ordered defaultdict list
    keys = [
        'dataset_id',
        

        'BMD', 'BMDL', 'BMDU', 'CSF',
        'AIC',
        'Chi2', 'df', 'residual_of_interest',

        'dfile', 'outfile',
    ]

    return {key: [] for key in keys}

flat = get_od()
# op = getattr(dichotomous_results[1].text(), 'BMD', {})

out = []

print(dichotomous_results[1])

for session in dichotomous_results:
    out.append([session.bmd,session.bmdl,session.bmdu])

print(out)
# output filename 
fn = os.path.expanduser(OUTPUT_FN)

output_df = pd.DataFrame(out)
#output_df.sort_values(['id'], inplace=True)
output_df.to_excel(fn, index=False)
#report.save("report.docx")

end_time = datetime.now()
delta = end_time - start_time

print(end_time)

print(delta)



#ContinuousModelSettings(settings = {"disttype": DistType.normal_ncv})

#create a continuous dataset
# dataset = ContinuousDataset(name="Body Weight from ChemX Exposure",
#     dose_units="ppm",
#     response_units="kg",
#     doses=[0, 25, 50, 75, 100],
#     ns=[20, 20, 20, 20, 20],
#     means=[6, 8, 13, 25, 30],
#     stdevs=[4, 4.3, 3.8, 4.4, 3.7]
# )



# create a dichotomous dataset
# dataset = DichotomousDataset(name="ChemX Nasal Lesion Incidence",
#     dose_units="ppm",
#     doses=[0, 25, 75, 125, 200],
#     ns=[20, 20, 20, 20, 20],
#     incidences=[0, 1, 7, 15, 19],
# )

# # session1 = bmds.BMDS.latest_version(dataset=dataset)
# # session1.add_default_bayesian_models(global_settings = {"bmr": 0.05, "bmr_type": DichotomousRiskType.AddedRisk, "alpha": 0.1})
# # session1.execute()

# # res = session1.model_average.results
# # print(f"BMD = {res.bmd:.2f} [{res.bmdl:.2f}, {res.bmdu:.2f}]")

# # bma_plot = session1.model_average.plot()
# # bma_plot.savefig("bma.png")


# dplot = dataset.plot()
# dplot.savefig("dataset-plot.png")



# # model = continuous.Hill(dataset)#, settings = {"disttype": DistType.normal_ncv, "bmr_type": ContinuousRiskType.RelativeDeviation})

# # print(model.settings.priors.tbl())

# # n = model.settings.priors.get_prior('n')
# # n.initial_value = 1
# # n.min_value = 1
# # n.max_value = 1
# # model.execute()
# # text = model.text()
# # print(text)

# # mplot = model.plot()
# # mplot.savefig("hill-plot.png")

# model = dichotomous.Logistic(dataset)#, settings = {"disttype": DistType.normal_ncv, "bmr_type": ContinuousRiskType.RelativeDeviation})

# print(model.settings.priors.tbl())

# a = model.settings.priors.get_prior('a')
# a.initial_value = 0
# a.min_value = -10
# a.max_value = 10
# b = model.settings.priors.get_prior('b')
# b.initial_value = 0
# b.min_value = -10
# b.max_value = 10
# model.execute()
# text = model.text()
# print(text)

# mplot = model.plot()
# mplot.savefig("logistic-plot.png")






### This is to run all models and select the best one
# create a BMD session
# session = bmds.BMDS.latest_version(dataset=dataset)

# # session.add_default_models(global_settings = {"bmr": 0.15, "bmr_type": DichotomousRiskType.AddedRisk, "alpha": 0.1})
# # session.add_default_models(global_settings = {"disttype": DistType.normal_ncv, "bmr_type": ContinuousRiskType.RelativeDeviation})
# session.add_default_models()

# # execute the session
# session.execute()

# # recommend a best-fitting model
# session.recommend()


# model_index = session.recommender.results.recommended_model_index
# if model_index:
#     model = session.models[model_index]
#     print(model.text())
#     mplot = model.plot()
#     mplot.savefig("recommended-model-plot.png")


# # save excel report
# df = session.to_df()
# df.to_excel("report.xlsx")

# # save to a word report
# report = session.to_docx()
# report.save("report.docx")

# # plot = session.model_average.plot(True)
# # plot.savefig("plot2.png")

# def plot(colorize: bool = False):
#     """
#     After model execution, print the dataset, curve-fit, BMD, and BMDL.
#     """
   
#     dataset = session.dataset
#     results = session.execute()
#     fig = dataset.plot()
#     ax = fig.gca()
#     ax.set_ylim(0, 35)
#     title = f"{dataset._get_dataset_name()}\nContinuous Models, 1 Standard Deviation"
#     ax.set_title(title)
#     if colorize:
#         color_cycle = cycle(plotting.INDIVIDUAL_MODEL_COLORS)
#         line_cycle = cycle(plotting.INDIVIDUAL_LINE_STYLES)
#     else:
#         color_cycle = cycle(["#ababab"])
#         line_cycle = cycle(["solid"])
#     for i, model in enumerate(session.models):
#         if colorize:
#             label = model.name()
#             print(label)
#         elif i == 0:
#             label = "Individual models"
#         else:
#             label = None
#         ax.plot(
#             model.results.plotting.dr_x,
#             model.results.plotting.dr_y,
#             #model.results.plotting.bmd_y,
#             label=label,
#             c=next(color_cycle),
#             linestyle=next(line_cycle),
#             zorder=100,
#             lw=2,
#         )
#         #model.results.plotting.add_bmr_lines(ax, model.results.plotting.bmd, model.results.plotting.bmd_y, model.results.plotting.bmdl, model.results.plotting.bmdu)
#     # ax.plot(
#     #     model.results.plotting.dr_x,
#     #     model.results.plotting.dr_y,
#     #     label="Model average (BMD, BMDL, BMDU)",
#     #     c="#6470C0",
#     #     lw=4,
#     #     zorder=110,
#     # )
#     # plotting.add_bmr_lines(ax, model.results.plotting.bmd, model.results.plotting.bmd_y, model.results.plotting.bmdl, model.results.plotting.bmdu)

#     # reorder handles and labels
#     handles, labels = ax.get_legend_handles_labels()
#     order = [2, 0, 1]
#     ax.legend(
#         #[handles[idx] for idx in order], [labels[idx] for idx in order], **plotting.LEGEND_OPTS
#     )

#     return fig


# plot = plot(True)
# plot.savefig("continuous-dataset.png")





