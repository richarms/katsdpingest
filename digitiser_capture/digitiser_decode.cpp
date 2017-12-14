#include <spead2/recv_udp_pcap.h>
#include <spead2/recv_stream.h>
#include <spead2/recv_ring_stream.h>
#include <spead2/recv_heap.h>
#include <spead2/common_logging.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <vector>
#include <limits>
#include <tbb/pipeline.h>
#include <tbb/task_scheduler_init.h>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

#if !SPEAD2_USE_PCAP
# error "spead2 was built without pcap support"
#endif

namespace po = boost::program_options;

#ifndef __BYTE_ORDER__
# warning "Unable to detect byte order"
#elif __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
# error "Only little endian is currently supported"
#endif

/***************************************************************************/

/* Take buffer of packed 10-bit signed values (big-endian) and return them as 16-bit
 * values.
 */
static std::vector<std::int16_t> decode_10bit(const std::uint8_t *data, std::size_t length, bool non_icd)
{
    std::size_t out_length = length * 8 / 10;
    std::vector<std::int16_t> out;
    out.reserve(out_length);
    std::vector<std::uint8_t> data2;
    if (non_icd)
    {
        /* Non-compliant bit packing. To fix it up:
         * - take 320 bits (40 bytes)
         * - split it into 64-bit values, and reverse them
         * - split it into 80-bit values, and reverse them
         */
        data2.resize(length);
        for (std::size_t i = 0; i < length; i += 40)
        {
            char shuffle[40];
            for (int j = 0; j < 40; j += 8)
                std::memcpy(&shuffle[32 - j], &data[i + j], 8);
            for (int j = 0; j < 40; j += 10)
                memcpy(&data2[i + j], &shuffle[30 - j], 10);
        }
        data = data2.data();
    }
    std::uint64_t buffer = 0;
    int buffer_bits = 0;
    for (std::size_t i = 0; i < length; i += 4)
    {
        std::uint32_t chunk;
        std::memcpy(&chunk, &data[i], 4);
        chunk = ntohl(chunk);
        buffer = (buffer << 32) | chunk;
        buffer_bits += 32;
        while (buffer_bits >= 10)
        {
            buffer_bits -= 10;
            std::int64_t value = (buffer >> buffer_bits) & 1023;
            // Convert to signed
            if (value & 512)
                value -= 1024;
            out.push_back(value);
        }
    }
    return out;
}

/***************************************************************************/

struct options
{
    bool non_icd = false;
    std::uint64_t max_heaps = std::numeric_limits<std::uint64_t>::max();
    std::string input_file;
    std::string output_file;
};

class heap_info
{
public:
    spead2::recv::heap heap;
    std::uint64_t timestamp = 0;
    const std::uint8_t *data = nullptr;
    std::size_t length = 0;   // length of payload in bytes
    std::size_t samples = 0;  // length in digitiser samples

    explicit heap_info(spead2::recv::heap &&heap);
    heap_info &operator=(spead2::recv::heap &&heap);

private:
    void update();
};

heap_info::heap_info(spead2::recv::heap &&heap) : heap(std::move(heap))
{
    update();
}

heap_info &heap_info::operator=(spead2::recv::heap &&heap)
{
    this->heap = std::move(heap);
    update();
    return *this;
}

void heap_info::update()
{
    timestamp = 0;
    data = nullptr;
    length = 0;
    samples = 0;
    for (const auto &item : heap.get_items())
    {
        if (item.id == 0x1600 && item.is_immediate)
            timestamp = item.immediate_value;
        else if (item.id == 0x3300)
        {
            data = item.ptr;
            length = item.length;
            samples = length * 8 / 10;
        }
    }
}

typedef std::vector<heap_info> heap_info_batch;
typedef std::vector<std::vector<std::int16_t>> decoded_batch;

class loader
{
private:
    spead2::thread_pool thread_pool;
    spead2::recv::ring_stream<> stream;
    // Buffer for heaps that were read while looking for sync, but still need
    // to be processed
    std::deque<heap_info> infoq;
    std::uint64_t next_timestamp;
    std::uint64_t max_heaps;
    bool finished = false;

public:
    std::uint64_t n_heaps = 0;
    std::uint64_t first_timestamp = 0;

    explicit loader(const options &opts)
        : thread_pool(),
        stream(thread_pool, 2, 128), max_heaps(opts.max_heaps)
    {
        stream.emplace_reader<spead2::recv::udp_pcap_file_reader>(opts.input_file);

        try
        {
            /* Two multicast groups are interleaved, and the multicast
             * subscriptions may have kicked in at different times. Proceed
             * until we see two consecutive samples.
             */
            infoq.emplace_back(stream.pop());
            infoq.emplace_back(stream.pop());
            std::cout << "First timestamp is " << infoq[0].timestamp << '\n';
            while (infoq[0].timestamp == 0 || infoq[1].timestamp == 0
                   || infoq[1].timestamp != infoq[0].timestamp + infoq[0].samples)
            {
                infoq.pop_front();
                infoq.emplace_back(stream.pop());
            }
            next_timestamp = infoq[0].timestamp;
            first_timestamp = next_timestamp;
            std::cout << "First synchronised timestamp is " << first_timestamp << '\n';
        }
        catch (spead2::ringbuffer_stopped)
        {
            throw std::runtime_error("End of stream reached before stream synchronisation");
        }
    }

    // Returns empty batch on reaching the end
    heap_info_batch next_batch()
    {
        constexpr int batch_size = 32;
        heap_info_batch batch;
        if (!finished)
        {
            for (int i = 0; i < batch_size; i++)
            {
                if (n_heaps >= max_heaps)
                {
                    std::cout << "Stopping after " << max_heaps << " heaps\n";
                    finished = true;
                    break;
                }
                if (infoq.empty())
                {
                    try
                    {
                        infoq.emplace_back(stream.pop());
                    }
                    catch (spead2::ringbuffer_stopped)
                    {
                        std::cout << "Stream ended after " << n_heaps << " heaps\n";
                        finished = true;
                        break;
                    }
                }
                heap_info info = std::move(infoq[0]);
                infoq.pop_front();
                if (info.timestamp != next_timestamp)
                {
                    std::cout.flush();
                    std::cerr << "Timestamps do not match up, aborting\n"
                        << "Expected " << next_timestamp << ", have " << info.timestamp << '\n';
                    finished = true;
                    break;
                }
                n_heaps++;
                next_timestamp += info.samples;
                batch.push_back(std::move(info));
            }
        }
        return batch;
    }
};

template<typename T>
static po::typed_value<T> *make_opt(T &var)
{
    return po::value<T>(&var)->default_value(var);
}

static po::typed_value<bool> *make_opt(bool &var)
{
    return po::bool_switch(&var)->default_value(var);
}

static void usage(std::ostream &o, const po::options_description &desc)
{
    o << "Usage: digitiser_decode [opts] <input.pcap> <output.npy>\n";
    o << desc;
}

static options parse_options(int argc, char **argv)
{
    options opts;
    po::options_description desc, hidden, all;
    desc.add_options()
        ("non-icd", make_opt(opts.non_icd), "Assume digitiser is not ICD compliant")
        ("heaps", make_opt(opts.max_heaps), "Number of heaps to process [all]")
    ;
    hidden.add_options()
        ("input", make_opt(opts.input_file), "input")
        ("output", make_opt(opts.output_file), "output")
    ;
    all.add(desc);
    all.add(hidden);

    po::positional_options_description positional;
    positional.add("input", 1);
    positional.add("output", 1);
    try
    {
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv)
            .style(po::command_line_style::default_style & ~po::command_line_style::allow_guessing)
            .options(all)
            .positional(positional)
            .run(), vm);
        po::notify(vm);
        if (vm.count("help"))
        {
            usage(std::cout, desc);
            std::exit(0);
        }
        return opts;
    }
    catch (po::error &e)
    {
        std::cerr << e.what() << '\n';
        usage(std::cerr, desc);
        std::exit(2);
    }
}

int main(int argc, char **argv)
{
    options opts = parse_options(argc, argv);
    // Leave 1 core free for decoding the SPEAD stream
    int n_threads = tbb::task_scheduler_init::default_num_threads() - 1;
    if (n_threads < 1)
        n_threads = 1;
    tbb::task_scheduler_init init_tbb(n_threads);

    loader load(opts);

    const int header_size = 96;
    std::ofstream out(opts.output_file, std::ios::out | std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    // Make space for the header
    out.seekp(header_size);
    std::uint64_t n_elements = 0;

    auto read_filter = [&] (tbb::flow_control &fc) -> std::shared_ptr<heap_info_batch>
    {
        std::shared_ptr<heap_info_batch> batch = std::make_shared<heap_info_batch>(load.next_batch());
        if (batch->empty())
            fc.stop();
        return batch;
    };

    auto decode_filter = [&](std::shared_ptr<heap_info_batch> batch) -> std::shared_ptr<decoded_batch>
    {
        std::shared_ptr<decoded_batch> out = std::make_shared<decoded_batch>();
        for (const heap_info &info : *batch)
            out->push_back(decode_10bit(info.data, info.length, opts.non_icd));
        return out;
    };

    auto write_filter = [&](std::shared_ptr<decoded_batch> batch)
    {
        for (const std::vector<int16_t> &decoded : *batch)
        {
            out.write(reinterpret_cast<const char *>(decoded.data()),
                      decoded.size() * sizeof(decoded[0]));
            n_elements += decoded.size();
        }
    };

    tbb::parallel_pipeline(16,
        tbb::make_filter<void, std::shared_ptr<heap_info_batch>>(
            tbb::filter::serial_in_order, read_filter)
        & tbb::make_filter<std::shared_ptr<heap_info_batch>, std::shared_ptr<decoded_batch>>(
            tbb::filter::parallel, decode_filter)
        & tbb::make_filter<std::shared_ptr<decoded_batch>, void>(
            tbb::filter::serial_in_order, write_filter));

    // Write in the header
    out.seekp(0);
    char header_start[10] = "\x93NUMPY\x01\x00";
    header_start[8] = header_size - 10;
    header_start[9] = 0;
    out.write(header_start, 10);
    out << "{'descr': '<i2', 'fortran_order': False, 'shape': ("
        << n_elements << ",) }";
    if (out.tellp() >= header_size)
    {
        std::cerr << "Oops, header was too big for reserved space! File is corrupted!\n";
        return 1;
    }
    while (out.tellp() < header_size - 1)
        out << ' ';
    out << '\n';
    out.close();
    std::cout << "Header successfully written\n";

    // Write the timestamp file
    std::ofstream timestamp_file(opts.output_file + ".timestamp");
    timestamp_file.exceptions(std::ios::failbit | std::ios::badbit);
    timestamp_file << load.first_timestamp << '\n';
    timestamp_file.close();
    std::cout << "Timestamp file written\n\n";
    std::cout << "Completed capture+conversion of " << load.n_heaps
        << " heaps from timestamp " << load.first_timestamp << '\n';
    return 0;
}
