# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_nn.test_layers import _dropout_helper


# Cosine Similarity
@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.cosine_similarity",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_value=2,
        max_value=5,
        min_dim_size=2,
        shape=(4, 4),
    ),
    axis=st.integers(min_value=-1, max_value=1),
)
def test_paddle_cosine_similarity(
    *,
    d_type_and_x,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="paddle.nn.functional.common.dropout",
    dtype_x_noiseshape=_dropout_helper(),
    rate=helpers.floats(min_value=0, max_value=0.9),
    seed=helpers.ints(min_value=0, max_value=100),
    test_with_out=st.just(False),
)
def test_paddle_dropout(
    *,
    dtype_x_noiseshape,
    rate,
    seed,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    (x_dtype, x), noise_shape = dtype_x_noiseshape
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        x=x[0],
        rate=rate,
        noise_shape=noise_shape,
        seed=seed,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    frontend_ret = helpers.flatten_and_to_np(ret=frontend_ret)
    for u, v, w in zip(ret, frontend_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape
