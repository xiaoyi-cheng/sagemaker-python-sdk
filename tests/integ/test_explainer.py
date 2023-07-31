# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import codecs
import json
import os
import pytest

import tests.integ
import tests.integ.timeout

from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer, JSONSerializer, JSONLinesSerializer
from sagemaker.deserializers import ExplanationDeserializer, CSVDeserializer, JSONLinesDeserializer, JSONDeserializer
from sagemaker.utils import unique_name_from_base
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.explainer.clarify_explainer_config import (
    ClarifyExplainerConfig,
    ClarifyInferenceConfig,
    ClarifyShapConfig,
    ClarifyShapBaselineConfig,
)

from tests.integ import DATA_DIR


ROLE = "SageMakerRole"
INSTANCE_COUNT = 1
INSTANCE_TYPE = "ml.c5.xlarge"
TEST_CSV_DATA = "42,42,42,42,42,42,42"
TEST_JSON_DATA = "{\"features\":[25, 2, 226802, 1, 7, 4, 6, 3, 2, 1, 0, 0, 40, 37]}"
SHAP_BASELINE = "1,2,3,4,5,6,7"
XGBOOST_DATA_PATH = os.path.join(DATA_DIR, "xgboost_model")
LINEAR_LEARNER_DATA_PATH = os.path.join(DATA_DIR, "linear_learner")

CLARIFY_SHAP_BASELINE_CONFIG = ClarifyShapBaselineConfig(shap_baseline=SHAP_BASELINE)
CLARIFY_SHAP_CONFIG = ClarifyShapConfig(shap_baseline_config=CLARIFY_SHAP_BASELINE_CONFIG)
CLARIFY_EXPLAINER_CONFIG = ClarifyExplainerConfig(
    shap_config=CLARIFY_SHAP_CONFIG, enable_explanations="`true`"
)
EXPLAINER_CONFIG = ExplainerConfig(clarify_explainer_config=CLARIFY_EXPLAINER_CONFIG)


@pytest.yield_fixture(scope="module")
def endpoint_name(sagemaker_session):
    endpoint_name = unique_name_from_base("clarify-explainer-enabled-endpoint-integ")
    xgb_model_data = sagemaker_session.upload_data(
        path=os.path.join(XGBOOST_DATA_PATH, "xgb_model.tar.gz"),
        key_prefix="integ-test-data/xgboost/model",
    )

    xgb_image = image_uris.retrieve(
        "xgboost",
        sagemaker_session.boto_region_name,
        version="1",
        image_scope="inference",
    )

    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=endpoint_name, sagemaker_session=sagemaker_session, hours=2
    ):
        xgb_model = Model(
            model_data=xgb_model_data,
            image_uri=xgb_image,
            name=endpoint_name,
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )
        xgb_model.deploy(
            INSTANCE_COUNT,
            INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            explainer_config=EXPLAINER_CONFIG,
        )
        yield endpoint_name


def test_describe_explainer_config(sagemaker_session, endpoint_name):
    endpoint_desc = sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

    endpoint_config_desc = sagemaker_session.sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_desc["EndpointConfigName"]
    )
    assert endpoint_config_desc["ExplainerConfig"] == EXPLAINER_CONFIG._to_request_dict()


def test_invoke_explainer_enabled_endpoint(sagemaker_session, endpoint_name):
    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
    )
    assert response
    # Explainer enabled endpoint content-type should always be "application/json"
    assert response.get("ContentType") == "application/json"
    response_body_stream = response["Body"]
    try:
        response_body_json = json.load(codecs.getreader("utf-8")(response_body_stream))
        assert response_body_json
        assert response_body_json.get("explanations")
        assert response_body_json.get("predictions")
    finally:
        response_body_stream.close()


def test_invoke_endpoint_with_on_demand_explanations(sagemaker_session, endpoint_name):
    response = sagemaker_session.sagemaker_runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        EnableExplanations="`false`",
        Body=TEST_CSV_DATA,
        ContentType="text/csv",
        Accept="text/csv",
    )
    assert response
    response_body_stream = response["Body"]
    try:
        response_body_json = json.load(codecs.getreader("utf-8")(response_body_stream))
        assert response_body_json
        # no records are explained when EnableExplanations="`false`"
        assert response_body_json.get("explanations") == {}
        assert response_body_json.get("predictions")
    finally:
        response_body_stream.close()


@pytest.yield_fixture(scope="module")
def ll_endpoint_name(sagemaker_session):
    endpoint_name = unique_name_from_base("clarify-explainer-enabled-linear-learner-endpoint-integ")
    ll_model_data = sagemaker_session.upload_data(
        path=os.path.join(LINEAR_LEARNER_DATA_PATH, "ll_uci_adult_model.tar.gz"),
        key_prefix="integ-test-data/linear_learner/model",
    )

    ll_image = image_uris.retrieve(
        "linear-learner",
        sagemaker_session.boto_region_name,
        version="1",
        image_scope="inference",
    )

    with tests.integ.timeout.timeout_and_delete_endpoint_by_name(
        endpoint_name=endpoint_name, sagemaker_session=sagemaker_session, hours=2
    ):
        ll_model = Model(
            model_data=ll_model_data,
            image_uri=ll_image,
            name=endpoint_name,
            role=ROLE,
            sagemaker_session=sagemaker_session,
        )
        LL_SHAP_BASELINE = "25, 2, 226802, 1, 7, 4, 6, 3, 2, 1, 0, 0, 40, 37"
        LL_CLARIFY_SHAP_BASELINE_CONFIG = ClarifyShapBaselineConfig(shap_baseline=LL_SHAP_BASELINE)
        LL_CLARIFY_SHAP_CONFIG = ClarifyShapConfig(shap_baseline_config=LL_CLARIFY_SHAP_BASELINE_CONFIG)
        LL_CLARIFY_INFERENCE_CONFIG = ClarifyInferenceConfig(
            features_attribute="features",
            content_template="{\"features\":$features}",
            probability_attribute="score",
            label_attribute="predicted_label"
        )
        LL_CLARIFY_EXPLAINER_CONFIG = ClarifyExplainerConfig(
            shap_config=LL_CLARIFY_SHAP_CONFIG, enable_explanations="`true`", inference_config=LL_CLARIFY_INFERENCE_CONFIG
        )
        LL_EXPLAINER_CONFIG = ExplainerConfig(clarify_explainer_config=LL_CLARIFY_EXPLAINER_CONFIG)
        ll_model.deploy(
            INSTANCE_COUNT,
            INSTANCE_TYPE,
            endpoint_name=endpoint_name,
            explainer_config=LL_EXPLAINER_CONFIG,
        )
        yield endpoint_name


def test_predict_xgb_endpoint(sagemaker_session, endpoint_name):
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
        deserializer=ExplanationDeserializer(prediction_deserializer=CSVDeserializer())
    )
    response = predictor.predict(TEST_CSV_DATA)
    print(response)
    assert response


def test_predict_linear_leaner_endpoint(sagemaker_session, ll_endpoint_name):
    predictor = Predictor(
        endpoint_name=ll_endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONLinesSerializer(),
        deserializer=ExplanationDeserializer(prediction_deserializer=JSONLinesDeserializer()),
    )
    response = predictor.predict(TEST_JSON_DATA)
    print(response)
    assert response
