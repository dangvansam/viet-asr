#ifndef VIETASR_MODULES_VIETASR_TRANSDUCER_DECODER_H
#define VIETASR_MODULES_VIETASR_TRANSDUCER_DECODER_H

#include <memory>
#include <vector>

#include "modules/vietasr/transducer_model.h"
#include "modules/vietasr/transducer_search.h"
#include "vietasr/types.h"

namespace vietasr {

class TransducerDecoder final {
public:
    TransducerDecoder(std::shared_ptr<const TransducerModel> model,
                      const TransducerSearchConfig& config);

    Status Decode(const EncoderOutput& encoder_out, std::vector<int>* tokens) const;

private:
    std::shared_ptr<const TransducerModel> model_;
    TransducerSearchConfig config_;
};

}

#endif
