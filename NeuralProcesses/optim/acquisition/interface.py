from trieste.acquisition.interface import AcquisitionFunctionBuilder, Dataset, Tag, ProbabilisticModelType, Mapping, Optional, AcquisitionFunction, SingleModelAcquisitionBuilder



class CustomizedSingleModelAcquisitionBuilder(SingleModelAcquisitionBuilder):
    """
    Convenience acquisition function builder for an acquisition function (or component of a
    composite acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: Tag) -> AcquisitionFunctionBuilder[ProbabilisticModelType]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """

        class _Anon(AcquisitionFunctionBuilder[ProbabilisticModelType]):
            def __init__(
                self, single_builder: SingleModelAcquisitionBuilder[ProbabilisticModelType]
            ):
                self.single_builder = single_builder

            def prepare_acquisition_function(
                self,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
                **kwargs
            ) -> AcquisitionFunction:
                return self.single_builder.prepare_acquisition_function(
                    models[tag], dataset=None if datasets is None else datasets[tag], **kwargs
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
                **kwargs
            ) -> AcquisitionFunction:
                return self.single_builder.update_acquisition_function(
                    function, models[tag], dataset=None if datasets is None else datasets[tag], **kwargs
                )

            def __repr__(self) -> str:
                return f"{self.single_builder!r} using tag {tag!r}"

        return _Anon(self)


