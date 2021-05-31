from flask import Flask
from flask_restful import Api, Resource, reqparse
from catboost import CatBoostClassifier

# Model
model0_main = CatBoostClassifier()
model0_main.load_model("model0_main_v200108")

model0_otheroperators = CatBoostClassifier()
model0_otheroperators.load_model("model0_otheroperators_v200114")

app=Flask(__name__)
api=Api(app)

class Predict_model0_main(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("ed_working_industry")
        parser.add_argument("pd_sex")
        parser.add_argument("device_info")
        parser.add_argument("address_flat_gr")
        parser.add_argument("appl_rqst_hh")
        parser.add_argument("age_binning")
        parser.add_argument("income_binning")
        parser.add_argument("cmnd_new_old")
        parser.add_argument("requested_amount")
        parser.add_argument("avg_NOTPAY_DOB65_cpd_phone")
        parser.add_argument("time_btwn_appearance_bor_sending_appl_binning")
        parser.add_argument("ed_occupation_type")
        parser.add_argument("time_btwn_filling_sending_appl_binning")        
        args=parser.parse_args()
        X=[args["ed_working_industry"],args["pd_sex"],args["device_info"],args["address_flat_gr"],args["appl_rqst_hh"],args["age_binning"],args["income_binning"],args["cmnd_new_old"],args["requested_amount"],args["avg_NOTPAY_DOB65_cpd_phone"],args["time_btwn_appearance_bor_sending_appl_binning"],args["ed_occupation_type"],args["time_btwn_filling_sending_appl_binning"]]
        return model0_main.predict_proba(X)[1], 200

class Predict_model0_otheroperators(Resource):
    
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument("ed_working_industry")
        parser.add_argument("pd_sex")
        parser.add_argument("device_info")
        parser.add_argument("address_flat_gr")
        parser.add_argument("appl_rqst_hh")
        parser.add_argument("age_binning")
        parser.add_argument("income_binning")
        parser.add_argument("cmnd_new_old")
        parser.add_argument("requested_amount")
        parser.add_argument("avg_NOTPAY_DOB65_cpd_phone")
        parser.add_argument("time_btwn_appearance_bor_sending_appl_binning")
        parser.add_argument("ed_occupation_type")
        parser.add_argument("time_btwn_filling_sending_appl_binning")        
        args=parser.parse_args()
        X=[args["ed_working_industry"],args["pd_sex"],args["device_info"],args["address_flat_gr"],args["appl_rqst_hh"],args["age_binning"],args["income_binning"],args["cmnd_new_old"],args["requested_amount"],args["avg_NOTPAY_DOB65_cpd_phone"],args["time_btwn_appearance_bor_sending_appl_binning"],args["ed_occupation_type"],args["time_btwn_filling_sending_appl_binning"]]
        return model0_otheroperators.predict_proba(X)[1], 200
    
    
api.add_resource(Predict_model0_main,"/predict_model0_main/")
api.add_resource(Predict_model0_otheroperators,"/predict_model0_otheroperators/")

if __name__ == "__main__":
  app.run()
    
