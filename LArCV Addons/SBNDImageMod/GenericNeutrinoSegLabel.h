/**
 * \file GenericSegLabel.h
 *
 * \ingroup ImageMod
 *
 * \brief Class def header for a class GenericSegLabel
 *
 * @author cadams
 * \edit by jhenzerling
 */

/** \addtogroup ImageMod

    @{*/
#ifndef __GENERICSEGLABEL_H__
#define __GENERICSEGLABEL_H__

#include "larcv/core/Processor/ProcessBase.h"
#include "larcv/core/Processor/ProcessFactory.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/Particle.h"
#include "larcv/core/DataFormat/Voxel2D.h"

namespace larcv {

/**
   \class ProcessBase
   User defined class GenericSegLabel ... these comments are
   used to generate
   doxygen documentation!
*/
class GenericSegLabel : public ProcessBase {
 public:
  /// Default constructor
  GenericSegLabel(
      const std::string name = "GenericSegLabel");

  /// Default destructor
  ~GenericSegLabel() {}

  void configure(const PSet&);

  void initialize();

  bool process(IOManager& mgr);

  void finalize();

  Image2D seg_image_creator(const std::vector<Particle> & particles,
                            const ClusterPixel2D & clusters,
                            const ImageMeta & meta);

 private:

  std::string _cluster2d_producer;
  std::string _output_producer;
  std::string _particle_producer;

};

/**
   \class larcv::GenericSegLabelFactory
   \brief A concrete factory class for larcv::GenericSegLabel
*/
class GenericSegLabelProcessFactory
    : public ProcessFactoryBase {
 public:
  /// ctor
  GenericSegLabelProcessFactory() {
    ProcessFactory::get().add_factory("GenericSegLabel",
                                      this);
  }
  /// dtor
  ~GenericSegLabelProcessFactory() {}
  /// creation method
  ProcessBase* create(const std::string instance_name) {
    return new GenericSegLabel(instance_name);
  }
};
}

#endif
/** @} */  // end of doxygen group
