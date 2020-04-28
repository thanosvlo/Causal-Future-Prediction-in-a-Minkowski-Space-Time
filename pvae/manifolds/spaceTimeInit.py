import torch.nn
from typing import Tuple, Optional
from manifolds.spaceTimeMath import SpaceTimeManifold
import geoopt
from geoopt.geoopt.utils import size2shape, broadcast_shapes
from geoopt.geoopt.manifolds.base import Manifold, ScalingInfo

__all__ = ["SpaceTime", "SpaceTimeExact"]

_spacetime_doc = r"""
    Spacetime model, see more in :doc:`/extended/spacetime`.

    Parameters
    ----------
    c : float|tensor
        ball negative curvature

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


# noinspection PyMethodOverriding
class SpaceTime(Manifold):
    __doc__ = r"""{}

    See Also
    --------
    :class:`SpaceTimeExact`
    """.format(
        _spacetime_doc
    )

    ndim = 1
    reversible = False
    name = "SpaceTime"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, c=1.0):
        super().__init__()
        self.math = SpaceTimeManifold()
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.get_default_dtype()))

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        px = self.math.project(x, c=self.c, dim=dim)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        return True, None

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return self.math.dist(x, y, c=self.c, keepdim=keepdim, dim=dim)

    def dist2(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return self.math.dist(x, y, c=self.c, keepdim=keepdim, dim=dim) ** 2

    # def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
    #     return self.math.egrad2rgrad(x, u, c=self.c, dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        # always assume u is scaled properly
        approx = x + u
        return self.math.project(approx, c=self.c, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return self.math.project(x, c=self.c, dim=dim)

    def proju(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1
    ) -> torch.Tensor:
        if v is None:
            v = u
        return self.math.inner(x, u, v, c=self.c, keepdim=keepdim, dim=dim)

    def norm(
        self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return self.math.normalize_tan(x, u)

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1
    ) -> torch.Tensor:
        res = self.math.expmap(x, u, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return self.math.logmap(x, y, c=self.c, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1):
        return self.math.parallel_transport(x, y, v, c=self.c, dim=dim)

    def transp_follow_retr(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        y = self.retr(x, u, dim=dim)
        return self.transp(x, y, v, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def expmap_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.expmap(x, u, dim=dim, project=project)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def retr_transp(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.retr(x, u, dim=dim)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def mobius_add(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_add(x, y, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_sub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_sub(x, y, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_coadd(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_coadd(x, y, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_cosub(
        self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_cosub(x, y, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_scalar_mul(
        self, r: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_scalar_mul(r, x, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_pointwise_mul(
        self, w: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_pointwise_mul(w, x, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def mobius_matvec(
        self, m: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.mobius_matvec(m, x, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def geodesic(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return self.math.geodesic(t, x, y, c=self.c, dim=dim)

    @__scaling__(ScalingInfo(t=-1))
    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True
    ) -> torch.Tensor:
        res = self.math.geodesic_unit(t, x, u, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    def lambda_x(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return self.math.lambda_x(x, c=self.c, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return self.math.dist0(x, c=self.c, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, dim=-1, project=True) -> torch.Tensor:
        res = self.math.expmap0(u, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(1))
    def logmap0(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return self.math.logmap0(x, c=self.c, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return self.math.parallel_transport0(y, u, c=self.c, dim=dim)

    def transp0back(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return self.math.parallel_transport0back(y, u, c=self.c, dim=dim)

    def gyration(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return self.math.gyration(x, y, z, c=self.c, dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist2plane(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        a: torch.Tensor,
        *,
        dim=-1,
        keepdim=False,
        signed=False
    ) -> torch.Tensor:
        return self.math.dist2plane(
            x, p, a, dim=dim, c=self.c, keepdim=keepdim, signed=signed
        )

    # this does not yet work with scaling
    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply(
        self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs
    ) -> torch.Tensor:
        res = self.math.mobius_fn_apply(fn, x, *args, c=self.c, dim=dim, **kwargs)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    # this does not yet work with scaling
    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply_chain(
        self, x: torch.Tensor, *fns: callable, project=True, dim=-1
    ) -> torch.Tensor:
        res = self.math.mobius_fn_apply_chain(x, *fns, c=self.c, dim=dim)
        if project:
            return self.math.project(res, c=self.c, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(std=-1), "random")
    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        """
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random point on the PoincareBall manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        size = size2shape(*size)
        self._assert_check_shape(size, "x")
        if device is not None and device != self.c.device:
            raise ValueError(
                "`device` does not match the manifold `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.c.dtype:
            raise ValueError(
                "`dtype` does not match the manifold `dtype`, set the `dtype` argument to None"
            )
        tens = (
            torch.randn(size, device=self.c.device, dtype=self.c.dtype)
            * std
            / size[-1] ** 0.5
            + mean
        )
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    random = random_normal

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        return geoopt.ManifoldTensor(
            torch.zeros(*size, dtype=dtype, device=device), manifold=self
        )


class SpaceTimeExact(SpaceTime):
    __doc__ = r"""{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`PoincareBall`
    """.format(
        _spacetime_doc
    )

    reversible = True
    retr_transp = SpaceTime.expmap_transp
    transp_follow_retr = SpaceTime.transp_follow_expmap
    retr = SpaceTime.expmap

    def extra_repr(self):
        return "exact"