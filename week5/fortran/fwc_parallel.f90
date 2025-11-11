!
! Applied High Performance Computing
! 
! Shallow Waters on GPUs
! 
! Assignment: Make a GPU parallelised shallow water code using OpenACC
!
! Author: Troels Haugbølle, Niels Bohr Institute, University of Copenhagen
! Date:   February 2022
! License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
!

module fwc_parallel_module
    use mpi
    use hdf5
    implicit none
    
    integer, parameter :: dp = kind(1.0e0_8) ! Double precision kind
    integer :: mpi_size, mpi_rank
    
    ! Representation of a flat world
    type :: World_t
        ! The current world time of the world
        real(dp) :: time
        ! The size of the world in the latitude dimension and the global size
        integer :: latitude, global_latitude
        ! The size of the world in the longitude dimension
        integer :: longitude, global_longitude
        ! The offset for this rank in the latitude dimension
        integer :: offset_latitude
        ! The offset for this rank in the longitude dimension
        integer :: offset_longitude
        ! The temperature of each coordinate of the world.
        ! Note: indexed as data(longitude, latitude)
        real(dp), allocatable :: data(:,:)
        ! The measure of the diffuse reflection of solar radiation at each world coordinate.
        ! See: <https://en.wikipedia.org/wiki/Albedo>
        ! Note: indexed as albedo_data(longitude, latitude)
        real(dp), allocatable :: albedo_data(:,:)
    end type World_t
    
contains

    ! Create a new flat world.
    !
    ! @param latitude     The size of the world in the latitude dimension.
    ! @param longitude    The size of the world in the longitude dimension.
    ! @param temperature  The initial temperature (the whole world starts with the same temperature).
    ! @param albedo_data  The measure of the diffuse reflection of solar radiation at each world coordinate.
    !                     This array must have shape (longitude, latitude).
    !
    function create_world(latitude, longitude, temperature, albedo_data_in) result(world)
        integer, intent(in) :: latitude, longitude
        real(dp), intent(in) :: temperature
        real(dp), intent(in) :: albedo_data_in(:,:)
        type(World_t) :: world
        
        world%latitude = latitude
        world%longitude = longitude
        world%time = 0.0_dp
        world%offset_latitude = 0
        world%offset_longitude = 0
        world%global_latitude = latitude
        world%global_longitude = longitude
        
        allocate(world%data(longitude, latitude))
        allocate(world%albedo_data(longitude, latitude))
        
        world%data = temperature
        world%albedo_data = albedo_data_in
    end function create_world

    ! Calculate checksum of the world
    ! Only loop *inside* data region -- not in ghostzones!
    !
    function checksum(world) result(cs)
        type(World_t), intent(in) :: world
        real(dp) :: cs
        integer :: i, j
        
        cs = 0.0_dp
        do j = 2, world%latitude - 1
            do i = 2, world%longitude - 1
                cs = cs + world%data(i, j)
            end do
        end do
    end function checksum

    ! Print statistics of the world
    !
    subroutine stat(world)
        type(World_t), intent(in) :: world
        real(dp) :: mint, maxt, meant
        integer  :: i, j
        
        mint = 1e99_dp
        maxt = 0.0_dp
        meant = 0.0_dp
        
        do j = 2, world%latitude - 1
            do i = 2, world%longitude - 1
                mint = min(mint, world%data(i, j))
                maxt = max(maxt, world%data(i, j))
                meant = meant + world%data(i, j)
            end do
        end do
        
        meant = meant / (world%global_latitude * world%global_longitude)
        
        write(*, '(A, F12.6, A, F12.6, A, F12.6)') &
            'min: ', mint, ', max: ', maxt, ', avg: ', meant
    end subroutine stat

    ! Exchange the ghost cells i.e. copy the second data row and column to the very last data row and column and vice versa.
    !
    ! @param world  The world to fix the boundaries for.
    !
    subroutine exchange_ghost_cells(world)
        type(World_t), intent(inout) :: world
        integer :: i, j
        
        ! Exchange columns (left and right boundaries)
        do j = 1, world%latitude
            world%data(1, j) = world%data(world%longitude - 1, j)
            world%data(world%longitude, j) = world%data(2, j)
        end do
        
        ! Exchange rows (top and bottom boundaries)
        do i = 1, world%longitude
            world%data(i, 1) = world%data(i, world%latitude - 1)
            world%data(i, world%latitude) = world%data(i, 2)
        end do
    end subroutine exchange_ghost_cells

    ! Warm the world based on the position of the sun.
    !
    ! @param world      The world to warm.
    !
    subroutine radiation(world)
        type(World_t), intent(inout) :: world
        real(dp) :: sun_angle, sun_intensity
        real(dp) :: sun_long, sun_lat, sun_height, sun_height_squared
        real(dp) :: delta_lat, delta_long, dist
        integer :: i, j
        
        sun_angle = cos(world%time)
        sun_intensity = 865.0_dp
        sun_long = (sin(sun_angle) * (world%global_longitude / 2.0_dp)) &
                   + world%global_longitude / 2.0_dp
        sun_lat = world%global_latitude / 2.0_dp
        sun_height = 100.0_dp + cos(sun_angle) * 100.0_dp
        sun_height_squared = sun_height * sun_height
        
        do j = 2, world%latitude - 1
            do i = 2, world%longitude - 1
                ! Euclidean distance between the sun and each earth coordinate
                delta_lat = sun_lat - ((j - 1) + world%offset_latitude)
                delta_long = sun_long - ((i - 1) + world%offset_longitude)
                dist = sqrt(delta_lat*delta_lat + delta_long*delta_long + sun_height_squared)
                world%data(i, j) = world%data(i, j) + &
                    (sun_intensity / dist) * (1.0_dp - world%albedo_data(i, j))
            end do
        end do
        
        call exchange_ghost_cells(world)
    end subroutine radiation

    ! Heat radiated to space
    !
    ! @param world  The world to update.
    !
    subroutine energy_emmision(world)
        type(World_t), intent(inout) :: world
        integer :: i, j
        
        do j = 1, world%latitude
            do i = 1, world%longitude
                world%data(i, j) = world%data(i, j) * 0.99_dp
            end do
        end do
    end subroutine energy_emmision

    ! Heat diffusion
    !
    ! @param world  The world to update.
    !
    subroutine diffuse(world)
        type(World_t), intent(inout) :: world
        real(dp), allocatable :: tmp(:,:)
        real(dp) :: center, left, right, up, down
        integer  :: k, i, j
        
        allocate(tmp(world%longitude, world%latitude))
        tmp = world%data
        
        do k = 1, 10
            do j = 2, world%latitude - 1
                do i = 2, world%longitude - 1
                    ! 5 point stencil
                    center = world%data(i, j)
                    left = world%data(i-1, j)
                    right = world%data(i+1, j)
                    up = world%data(i, j-1)
                    down = world%data(i, j+1)
                    tmp(i, j) = (center + left + right + up + down) / 5.0_dp
                end do
            end do
            
            ! Swap arrays
            call move_alloc(from=tmp, to=world%data)
            allocate(tmp(world%longitude, world%latitude))
            tmp = world%data
            
            call exchange_ghost_cells(world)
        end do
        
        deallocate(tmp)
    end subroutine diffuse

    ! One integration step at `world_time`
    !
    ! @param world      The world to update.
    !
    subroutine integrate(world)
        type(World_t), intent(inout) :: world
        
        call radiation(world)
        call energy_emmision(world)
        call diffuse(world)
    end subroutine integrate

    ! Read a world model from a HDF5 file
    !
    ! @param filename The path to the HDF5 file.
    ! @return         A new world based on the HDF5 file.
    !
    function read_world_model(filename) result(world)
        character(len=*), intent(in) :: filename
        type(World_t) :: world
        
        integer(hid_t) :: file_id, dataset_id, dataspace_id
        integer(hsize_t) :: dims(2), maxdims(2)
        integer :: ndims
        real(dp), allocatable :: data_flat(:)
        real(dp), allocatable :: data_2d(:,:)
        integer :: latitude, longitude, i, j
        integer :: ierr
        
        ! Initialize HDF5
        call h5open_f(ierr)
        
        ! Open the HDF5 file in read-only mode
        call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, ierr)
        if (ierr /= 0) then
            print *, "Error: Could not open HDF5 file: ", trim(filename)
            stop
        end if
        
        ! Open the dataset
        call h5dopen_f(file_id, "world", dataset_id, ierr)
        if (ierr /= 0) then
            print *, "Error: Could not open 'world' dataset in HDF5 file"
            call h5fclose_f(file_id, ierr)
            stop
        end if
        
        ! Get the dataspace to check dimensions
        call h5dget_space_f(dataset_id, dataspace_id, ierr)
        
        ! Get the dimensions
        call h5sget_simple_extent_ndims_f(dataspace_id, ndims, ierr)
        if (ndims /= 2) then
            print *, "Error: Dataset must have 2 dimensions"
            call h5sclose_f(dataspace_id, ierr)
            call h5dclose_f(dataset_id, ierr)
            call h5fclose_f(file_id, ierr)
            stop
        end if
        
        call h5sget_simple_extent_dims_f(dataspace_id, dims, maxdims, ierr)
        ! dims(1) = longitude, dims(2) = latitude in C-order storage
        longitude = int(dims(1))
        latitude = int(dims(2))
        
        ! Allocate arrays
        allocate(data_flat(longitude * latitude))
        allocate(data_2d(longitude, latitude))
        
        ! Read the data as 1D
        call h5dread_f(dataset_id, H5T_NATIVE_DOUBLE, data_flat, dims, ierr)
        if (ierr /= 0) then
            print *, "Error: Could not read data from HDF5 file"
            deallocate(data_2d)
            deallocate(data_flat)
            call h5sclose_f(dataspace_id, ierr)
            call h5dclose_f(dataset_id, ierr)
            call h5fclose_f(file_id, ierr)
            stop
        end if
        
        ! Convert from 1D to 2D array (longitude, latitude)
        do j = 1, latitude
            do i = 1, longitude
                data_2d(i, j) = data_flat((j-1)*longitude + i)
            end do
        end do
        
        ! Create the world with the 2D data
        world = create_world(latitude, longitude, 293.15_dp, data_2d)
        
        write(*,'(A,I0,A,I0)') "World model loaded -- latitude: ", latitude, ", longitude: ", longitude
        
        ! Clean up
        deallocate(data_2d)
        deallocate(data_flat)
        call h5sclose_f(dataspace_id, ierr)
        call h5dclose_f(dataset_id, ierr)
        call h5fclose_f(file_id, ierr)
        call h5close_f(ierr)
        
    end function read_world_model

    ! Write data to a hdf5 file
    !
    ! @param world      The world to write
    ! @param filename   The output filename of the HDF5 file
    ! @param iteration  The iteration number
    !
    subroutine write_hdf5(world, filename, iteration)
        type(World_t),      intent(in) :: world
        character(len=*), intent(in) :: filename
        integer,          intent(in) :: iteration
        
        integer(hid_t) :: file_id, group_id, dataset_id, dataspace_id
        integer(hsize_t) :: dims(2)
        character(len=32) :: group_name
        real(dp), allocatable :: data_flat(:)
        integer :: i, j, ierr
        logical :: file_exists
        logical, save :: hdf5_initialized = .false.
        
        ! Initialize HDF5 library on first call
        if (.not. hdf5_initialized) then
            call h5open_f(ierr)
            if (ierr /= 0) then
                print *, "Error: Could not initialize HDF5 library"
                return
            end if
            hdf5_initialized = .true.
        end if
        
        ! Check if file exists to determine access mode
        inquire(file=trim(filename), exist=file_exists)
        
        ! HDF5 dimensions in C-order: dims(1)=latitude, dims(2)=longitude
        ! This matches the C++ code: {world.latitude, world.longitude}
        dims(1) = world%latitude
        dims(2) = world%longitude
        
        ! Convert from Fortran 2D (longitude, latitude) to C 1D (latitude, longitude) layout
        ! C storage: data[lat][lon] means index = lat * longitude + lon
        allocate(data_flat(world%longitude * world%latitude))
        do j = 1, world%latitude
            do i = 1, world%longitude
                ! C-order: row-major with latitude as rows, longitude as columns
                data_flat((j-1)*world%longitude + i) = world%data(i, j)
            end do
        end do
        
        ! Open or create the HDF5 file
        if (.not. file_exists) then
            call h5fcreate_f(filename, H5F_ACC_TRUNC_F, file_id, ierr)
        else
            call h5fopen_f(filename, H5F_ACC_RDWR_F, file_id, ierr)
        end if
        
        if (ierr /= 0) then
            print *, "Error: Could not open/create HDF5 file: ", trim(filename)
            deallocate(data_flat)
            return
        end if
        
        ! Create a group for this iteration
        write(group_name, '(I0)') iteration
        call h5gcreate_f(file_id, "/"//trim(group_name), group_id, ierr)
        
        if (ierr /= 0) then
            print *, "Error: Could not create group in HDF5 file"
            deallocate(data_flat)
            call h5fclose_f(file_id, ierr)
            return
        end if
        
        ! Create the dataspace
        call h5screate_simple_f(2, dims, dataspace_id, ierr)
        
        ! Create the dataset
        call h5dcreate_f(group_id, "world", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, ierr)
        
        ! Write the data from 1D array
        call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, data_flat, dims, ierr)
        
        ! Close the dataset, dataspace, and group
        call h5dclose_f(dataset_id, ierr)
        call h5sclose_f(dataspace_id, ierr)
        call h5gclose_f(group_id, ierr)
        call h5fclose_f(file_id, ierr)
        
        deallocate(data_flat)
    end subroutine write_hdf5

    ! Simulation of a flat word climate
    !
    ! @param num_of_iterations  Number of time steps to simulate
    ! @param model_filename     The filename of the world model to use (HDF5 file)
    ! @param output_filename    The filename of the written world history (HDF5 file)
    !
    subroutine simulate(num_of_iterations, model_filename, output_filename)
        integer, intent(in) :: num_of_iterations
        character(len=*), intent(in) :: model_filename
        character(len=*), intent(in) :: output_filename
        
        type(World_t) :: global_world, world
        integer     :: offset_longitude, offset_latitude
        integer     :: longitude, latitude
        real(dp), allocatable :: albedo(:,:)
        real(dp) :: delta_time
        integer  :: iteration, i, j, ig, jg
        integer  :: clock_start, clock_end, clock_rate
        real(dp) :: elapsed_time
        
        ! For simplicity, read in full model
        global_world = read_world_model(model_filename)
        
        ! Figure out size of domain for this rank
        offset_longitude = -1  ! -1 because first cell is a ghostcell
        offset_latitude = -1
        longitude = global_world%longitude + 2  ! one ghost cell on each end
        latitude = global_world%latitude + 2
        
        ! Copy over albedo data to local world data
        allocate(albedo(longitude, latitude))
        albedo = 0.0_dp
        do j = 2, latitude - 1
            do i = 2, longitude - 1
                ig = i + offset_longitude
                jg = j + offset_latitude
                albedo(i, j) = global_world%albedo_data(ig, jg)
            end do
        end do
        
        ! Create local world data
        world = create_world(latitude, longitude, 293.15_dp, albedo)
        world%global_latitude = global_world%global_latitude
        world%global_longitude = global_world%global_longitude
        world%offset_latitude = offset_latitude
        world%offset_longitude = offset_longitude
        
        ! Set up counters and loop for num_iterations of integration steps
        delta_time = world%global_longitude / 36.0_dp
        
        call system_clock(count_rate=clock_rate)
        call system_clock(count=clock_start)
        
        do iteration = 0, num_of_iterations - 1
            world%time = iteration / delta_time
            call integrate(world)
            
            ! Remove ghostzones and construct global data from local data
            if (len_trim(output_filename) > 0) then
                do j = 2, latitude - 1
                    do i = 2, longitude - 1
                        global_world%data(i - 1, j - 1) = world%data(i, j)
                    end do
                end do
                ! Only rank zero writes water history to file
                if (mpi_rank == 0) then
                    call write_hdf5(global_world, output_filename, iteration)
                    write(*, '(I0, A)', advance='no') iteration, ' -- '
                    call stat(global_world)
                end if
            end if
        end do
        
        call system_clock(count=clock_end)
        elapsed_time = real(clock_end - clock_start, dp) / real(clock_rate, dp)
        
        if (mpi_rank == 0) then
            call stat(world)
            write(*, '(A, E15.6)') 'checksum      : ', checksum(world)
            write(*, '(A, F15.6, A)') 'elapsed time  : ', elapsed_time, ' sec'
        end if
        
        deallocate(albedo)
        deallocate(world%data)
        deallocate(world%albedo_data)
        deallocate(global_world%data)
        deallocate(global_world%albedo_data)
    end subroutine simulate

end module fwc_parallel_module


program fwc_parallel
    use fwc_parallel_module
    use iso_fortran_env, only: error_unit
    implicit none
    
    integer :: iterations = 0
    character(len=256) :: model_filename = ''
    character(len=256) :: output_filename = ''
    integer :: i, argc, ierr
    character(len=256) :: arg, next_arg
    character(len=MPI_MAX_PROCESSOR_NAME) :: processor_name
    integer :: name_len
    
    ! Main function that parses the command line and start the simulation
    call MPI_Init(ierr)
    
    ! Get the number of processes
    call MPI_Comm_size(MPI_COMM_WORLD, mpi_size, ierr)
    ! Get the rank of the process
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, ierr)
    ! Print out the rank information
    call MPI_Get_processor_name(processor_name, name_len, ierr)
    if (mpi_rank == 0) then
        write(*,'(A,I0,A,I0,A,A)') "Running on ", mpi_size, " processes, this is rank ", &
            mpi_rank, " running on ", trim(processor_name)
    end if
    
    argc = command_argument_count()
    
    i = 1
    do while (i <= argc)
        call get_command_argument(i, arg)
        
        if (arg == '-h') then
            print *, './fwc --iter <number of iterations> --model <input model> --out <name of output file>'
            call MPI_Finalize(ierr)
            stop
        else if (i == argc) then
            write(error_unit, '(A)') 'Error: The last argument (' // trim(arg) // ') must have a value'
            call MPI_Finalize(ierr)
            stop 1
        end if
        
        if (arg == '--iter') then
            call get_command_argument(i + 1, next_arg)
            read(next_arg, *) iterations
            if (iterations < 0) then
                write(error_unit, '(A)') 'Error: iter must be positive (e.g. --iter 1000)'
                call MPI_Finalize(ierr)
                stop 1
            end if
            i = i + 2
        else if (arg == '--model') then
            call get_command_argument(i + 1, model_filename)
            i = i + 2
        else if (arg == '--out') then
            call get_command_argument(i + 1, output_filename)
            i = i + 2
        else
            write(error_unit, '(A)') 'Error: the argument type is not recognized'
            i = i + 1
        end if
    end do
    
    if (len_trim(model_filename) == 0) then
        write(error_unit, '(A)') 'Error: You must specify the model to simulate (e.g. --model ../models/small.hdf5)'
        call MPI_Finalize(ierr)
        stop 1
    end if
    if (iterations == 0) then
        write(error_unit, '(A)') 'Error: You must specify the number of iterations (e.g. --iter 10)'
        call MPI_Finalize(ierr)
        stop 1
    end if
    
    call simulate(iterations, model_filename, output_filename)
    
    call MPI_Finalize(ierr)

end program fwc_parallel
