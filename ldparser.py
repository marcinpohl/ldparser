""" Parser for MoTec ld files

Code created through reverse engineering the data format.
"""

import datetime
import struct
import mmap
from pathlib import Path

import numpy as np


class ldData:
    """Container for parsed data of ld file. Allows reading and writing. """

    def __init__(self, data_file: Path) -> None:
        self.data_file = data_file
        with open(self.data_file, mode='rb') as fd:
            with mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                ###TODO extract all the pointer structures, make a 'index table' out of them
                ###then make memoryviews out of each section, for clear fmt.unpack(mysection)

                ### ldHead
                self.head = ldHead(mm[0:ldHead.fmt_struct.size])
                ### ldEvent
                if self.head.aux_ptr > 0:
                    # self.head.aux_ptr is the venue offset
                    event_start = self.head.aux_ptr
                    event_end = event_start + ldEvent.fmt_struct.size
                    self.aux = ldEvent(mm[event_start:event_end])
                ### ldVenue
                if self.aux.venue_ptr > 0:
                    venue_start = self.aux.venue_ptr
                    venue_end = venue_start + ldVenue.fmt_struct.size
                    self.venue = ldVenue(mm[venue_start:venue_end])
                ### ldVehicle
                if self.venue.vehicle_ptr > 0:
                    vehicle_start = self.venue.vehicle_ptr
                    vehicle_end = vehicle_start + ldVehicle.fmt_struct.size
                    self.vehicle = ldVehicle(mm[vehicle_start:vehicle_end])
                ### ldChan
                self.channs = []  ###TODO dataclasses?
                ### TODO bring the rest of the sections here.
                ### pass in only the areas of mm that are needed
                ### probably keep track of offsets for sections
                ### keep converting the fromfile to proper constructors

    def read_channels(self, meta_ptr: int) -> list:
        """ Read channel data inside ld file

        Cycles through the channels inside ld file,
         starting with the one where meta_ptr points to.
         Returns a list of ldchan objects.
        """
        chans = []
        while meta_ptr:
            chan_ = ldChan.fromfile(self.data_file, meta_ptr)
            chans.append(chan_)
            meta_ptr = chan_.next_meta_ptr
        return chans

    def __getitem__(self, item):
        if not isinstance(item, int):
            col = [n for n, x in enumerate(self.channs) if x.name == item]
            if len(col) != 1:
                raise Exception("Could get column", item, col)
            item = col[0]
        return self.channs[item]

    def __iter__(self):
        return iter(x.name for x in self.channs)

    @classmethod
    def frompd(cls, df):
        raise NotImplementedError
        # type: (pd.DataFrame) -> ldData
        """Create and ldData object from a pandas DataFrame.

        Example:
        import pandas as pd
        import numpy as np
        from ldparser import ldData

        # create test dataframe
        df = pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
        print(df)
        # create an lddata object from the dataframe
        l = ldData.frompd(df)
        # write an .ld file
        l.write('/tmp/test.ld')

        # just to check, read back the file
        l = ldData.fromfile('/tmp/test.ld')
        # create pandas dataframe
        df = pd.DataFrame(data={c: l[c].data for c in l})
        print(df)

        """

        # for now, fix datatype and frequency
        ### TODO check Dennis vs Peter files, 60 vs 360Hz logs
        freq, dtype = 10, np.float32

        # pointer to meta data of first channel
        meta_ptr = ldHead.fmt_struct.size

        # list of columns to read - only accept numeric data
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

        # pointer to data of first channel
        ###TODO make a section offsets storage in general, not just frompd()
        data_ptr = meta_ptr + len(cols) * ldChan.fmt_struct.size

        # create a mocked header
        ###TODO needs major readjustment to fit the new constructor
        head = ldHead(meta_ptr, data_ptr, 0, None,
                      "testdriver", "testvehicleid", "testvenue",
                      datetime.datetime.now(),
                      "just a test", "testevent", "practice")

        ### TODO separate function
        # create the channels, meta data and associated data
        channs, prev, next = [], 0, meta_ptr + ldChan.fmt_struct.size
        for n, col in enumerate(cols):
            # create mocked channel header
            chan = ldChan(None,
                          meta_ptr, prev, next if n < len(cols) - 1 else 0,
                          data_ptr, len(df[col]),
                          dtype, freq, 0, 1, 1, 0,
                          col, col, "m")

            # link data to the channel
            chan._data = df[col].to_numpy(dtype)

            # calculate pointers to the previous/next channel meta data
            prev = meta_ptr
            meta_ptr = next
            next += ldChan.fmt_struct.size

            # increment data pointer for next channel
            data_ptr += chan._data.nbytes

            channs.append(chan)
        return cls(head, channs)

    def write(self, f: str) -> None:
        raise NotImplementedError
        """Write ld file containing the current header information and channel data """

        # convert the data using scale/shift etc before writing the data
        conv_data = lambda c: ((c.data / c.mul) - c.shift) * c.scale / pow(10., -c.dec)

        with open(f, 'wb') as f_:
            self.head.write(f_, len(self.channs))
            f_.seek(self.channs[0].meta_ptr)
            list(map(lambda c: c[1].write(f_, c[0]), enumerate(self.channs)))
            list(map(lambda c: f_.write(conv_data(c).astype(c.dtype)), self.channs))


class ldEvent:
    fmt = '<64s64s1024sH'
    fmt_struct = struct.Struct(fmt)
    del fmt

    def __init__(self, section_file_map):
        name, session, comment, self.venue_ptr = ldEvent.fmt_struct.unpack(section_file_map)
        self.name, self.session, self.comment = map(normalize_text, [name, session, comment])

    def write(self, f):
        raise NotImplementedError
        f.write(struct.pack(ldEvent.fmt,
                            self.name.encode(),
                            self.session.encode(),
                            self.comment.encode(),
                            self.venue_ptr))

        if self.venue_ptr > 0:
            f.seek(self.venue_ptr)
            self.venue.write(f)

    def __str__(self):
        return f"Event: {vars(self)}"


class ldVenue:
    fmt = '<64s1034xH'
    fmt_struct = struct.Struct(fmt)
    del fmt

    def __init__(self, section_file_map):
        """Parses and stores the venue information in ld file """
        name, self.vehicle_ptr = ldVenue.fmt_struct.unpack(section_file_map)
        self.name = normalize_text(name)

    def write(self, f):
        raise NotImplementedError
        f.write(struct.pack(ldVenue.fmt, self.name.encode(), self.vehicle_ptr))

        if self.vehicle_ptr > 0:
            f.seek(self.vehicle_ptr)
            self.vehicle.write(f)

    def __str__(self):
        return f"Venue: {vars(self)}"


class ldVehicle:
    fmt = '<64s128xI32s32s'
    fmt_struct = struct.Struct(fmt)
    del fmt

    def __init__(self, section_file_map):
        """Parses and stores the vehicle information in an ld file """
        # self.id, self.weight, self.type, self.comment = id, weight, type, comment
        vehicle_id, self.weight, vehicle_type, comment = ldVehicle.fmt_struct.unpack(section_file_map)
        self.id, self.type, self.comment = map(normalize_text, [vehicle_id, vehicle_type, comment])

    def write(self, f):
        raise NotImplementedError
        f.write(struct.pack(ldVehicle.fmt, self.id.encode(), self.weight, self.type.encode(), self.comment.encode()))

    def __str__(self):
        return f"Venue: {vars(self)}"


class ldHead:
    fmt = '<' + (
        "I4x"  # ldmarker
        "II"  # chann_meta_ptr chann_data_ptr
        "20x"  # ??
        "I"  # event_ptr
        "24x"  # ??
        "HHH"  # unknown static (?) numbers
        "I"  # device serial
        "8s"  # device type
        "H"  # device version
        "H"  # unknown static (?) number
        "I"  # num_channs
        "4x"  # ??
        "16s"  # date
        "16x"  # ??
        "16s"  # time
        "16x"  # ??
        "64s"  # driver
        "64s"  # vehicleid
        "64x"  # ??
        "64s"  # venue
        "64x"  # ??
        "1024x"  # ??
        "I"  # enable "pro logging" (some magic number?)
        "66x"  # ??
        "64s"  # short comment
        "126x"  # ??
        "64s"  # event
        "64s"  # session
    )
    fmt_struct = struct.Struct(fmt)
    del fmt

    def __init__(self, section_file_map) -> None:
        """Parses and stores the header information of ld file """
        (_, meta_ptr, data_ptr, aux_ptr,
         _, _, _, _, _, _, _,
         n, date, time,
         driver, vehicle_id, venue, _, short_comment, event, session) = ldHead.fmt_struct.unpack(section_file_map)

        most_entries = (date, time, driver, vehicle_id, venue, short_comment, event, session)
        date, time, driver, vehicle_id, venue, short_comment, event, session = [normalize_text(entry) for entry in
                                                                                most_entries]

        try:
            # first, try to decode datatime with seconds
            _datetime = datetime.datetime.strptime('%s %s' % (date, time), '%d/%m/%Y %H:%M:%S')
        except ValueError:
            _datetime = datetime.datetime.strptime('%s %s' % (date, time), '%d/%m/%Y %H:%M')

        ### populate it with dict(zip(a,b)) ?
        self.meta_ptr, self.data_ptr, self.aux_ptr, self.driver, self.vehicle_id, self.venue, self.datetime, self.short_comment, self.event, self.session = \
            meta_ptr, data_ptr, aux_ptr, driver, vehicle_id, venue, _datetime, short_comment, event, session

    def write(self, f, n):
        raise NotImplementedError
        f.write(struct.pack(ldHead.fmt,
                            0x40,
                            self.meta_ptr, self.data_ptr, self.aux_ptr,
                            1, 0x4240, 0xf,
                            0x1f44, "ADL".encode(), 420, 0xadb0, n,
                            self.datetime.date().strftime("%d/%m/%Y").encode(),
                            self.datetime.time().strftime("%H:%M:%S").encode(),
                            self.driver.encode(), self.vehicleid.encode(), self.venue.encode(),
                            0xc81a4, self.short_comment.encode(), self.event.encode(), self.session.encode(),
                            ))
        if self.aux_ptr > 0:
            f.seek(self.aux_ptr)
            self.aux.write(f)

    def __str__(self):
        return f"Header: {vars(self)}"


class ldChan:
    """Channel (meta) data

    Parses and stores the channel metadata in a ld file.
    Needs the pointer to the channel meta block in the ld file.
    The actual data is read on demand using the 'data' property.
    """

    fmt = '<' + (
        "IIII"  # prev_addr next_addr data_ptr n_data
        "H"  # some counter?
        "HHH"  # datatype datatype rec_freq
        "hhhh"  # shift mul scale dec_places
        "32s"  # name
        "8s"  # short name
        "12s"  # unit
        "40x"  # ? (40 bytes for ACC, 32 bytes for acti)
    )
    fmt_struct = struct.Struct(fmt)
    del fmt

    def __init__(self, _f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                 dtype, freq, shift, mul, scale, dec,
                 name, short_name, unit):

        self._f = _f
        self.meta_ptr = meta_ptr
        self._data = None

        (self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
         self.dtype, self.freq,
         self.shift, self.mul, self.scale, self.dec,
         self.name, self.short_name, self.unit) = prev_meta_ptr, next_meta_ptr, data_ptr, data_len, \
                                                  dtype, freq, \
                                                  shift, mul, scale, dec, \
                                                  name, short_name, unit

    @classmethod
    def fromfile(cls, _f, meta_ptr):
        # type: (str, int) -> ldChan
        """Parses and stores the header information of ld channel in a ld file """
        with open(_f, 'rb') as f:
            f.seek(meta_ptr)

            (prev_meta_ptr, next_meta_ptr, data_ptr, data_len, _,
             dtype_a, dtype, freq, shift, mul, scale, dec,
             name, short_name, unit) = struct.unpack(ldChan.fmt, f.read(struct.calcsize(ldChan.fmt)))

        name, short_name, unit = map(normalize_text, [name, short_name, unit])

        if dtype_a in [0x07]:
            dtype = [None, np.float16, None, np.float32][dtype - 1]
        elif dtype_a in [0, 0x03, 0x05]:
            dtype = [None, np.int16, None, np.int32][dtype - 1]
        else:
            raise TypeError('Datatype %i not recognized' % dtype_a)

        return cls(_f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len, dtype, freq, shift, mul, scale, dec,
                   name, short_name, unit)

    def write(self, f, n):
        raise NotImplementedError
        if self.dtype == np.float16 or self.dtype == np.float32:
            dtype_a = 0x07
            dtype = {np.float16: 2, np.float32: 4}[self.dtype]
        else:
            dtype_a = 0x05 if self.dtype == np.int32 else 0x03
            dtype = {np.int16: 2, np.int32: 4}[self.dtype]

        f.write(struct.pack(ldChan.fmt,
                            self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
                            0x2ee1 + n, dtype_a, dtype, self.freq, self.shift, self.mul, self.scale, self.dec,
                            self.name.encode(), self.short_name.encode(), self.unit.encode()))

    @property
    def data(self) -> np.array:
        """ Read the data words of the channel """
        if self._data is None:
            # jump to data and read
            with open(self._f, 'rb') as f:
                f.seek(self.data_ptr)
                try:
                    self._data = np.fromfile(f, count=self.data_len, dtype=self.dtype)
                    self._data = (self._data / self.scale * pow(10., -self.dec) + self.shift) * self.mul

                    if len(self._data) != self.data_len:
                        raise ValueError("Not all data read!")

                except ValueError as v:
                    print(v, self.name, self.freq,
                          hex(self.data_ptr), hex(self.data_len),
                          hex(len(self._data)), hex(f.tell()))
                    # raise v
        return self._data

    def __str__(self):
        ### TODO f-strings
        return 'chan %s (%s) [%s], %i Hz' % (
            self.name,
            self.short_name, self.unit,
            self.freq)


def normalize_text(inputs: bytes) -> str:
    """decode the bytes and remove trailing zeros """
    try:
        ###TODO do we want to remove everything that's not ASCII printable?
        return inputs.decode('ascii').strip().rstrip('\0').strip()
    except Exception as e:
        print("Could not decode string: %s - %s" % (e, inputs))
        return ""
        # raise e


if __name__ == '__main__':
    """ Small test of the parser.
    
    Decodes all ld files in the directory. For each file, creates 
    a plot for data with the same sample frequency.  
    """

    import sys
    from itertools import groupby
    import pandas as pd
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print("Usage: ldparser.py /some/path/somefile.ld")
        sys.exit(1)

    data_files = Path.cwd().glob('*.ld')
    for data_file in data_files:
        print(f" [*] Processing file: {data_file}")
        l = ldData(data_file)
        ###checking __str__
        print(f" [*] {l.head}")
        print(f" [*] {l.aux}")
        print(f" [*] {l.vehicle}")
        print(f" [*] {l.venue}")
        print(f" [*] {l.channs}")  ### TODO not done, just a control value for empties

        sys.exit(1)

        ###TODO make into a test case?  debug aid?
        # create plots for all channels with the same frequency
        for f, g in groupby(l.channs, lambda x: x.freq):
            df = pd.DataFrame({i.name.lower(): i.data for i in g})
            df.plot()
            plt.show()
