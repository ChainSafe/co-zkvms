use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use super::{
    PartyID,
};

/// This trait defines the network interface for the REP3 protocol.
pub trait Rep3Network: Send {
    /// Returns the id of the party. The id is in the range 0 <= id < 3
    fn get_id(&self) -> PartyID;

    /// Sends `data` to the next party and receives from the previous party. Use this whenever
    /// possible in contrast to calling [`Self::send_next()`] and [`Self::recv_prev()`] sequential. This method
    /// executes send/receive concurrently.
    fn reshare<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<F> {
        let mut res = self.reshare_many(&[data])?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            //we checked that there is really one element
            Ok(res.pop().unwrap())
        }
    }

    /// Perform multiple reshares with one networking round
    fn reshare_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<Vec<F>>;

    /// Broadcast data to the other two parties and receive data from them
    fn broadcast<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: F,
    ) -> std::io::Result<(F, F)> {
        let (mut prev, mut next) = self.broadcast_many(&[data])?;
        if prev.len() != 1 || next.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            //we checked that there is really one element
            let prev = prev.pop().unwrap();
            let next = next.pop().unwrap();
            Ok((prev, next))
        }
    }

    /// Broadcast data to the other two parties and receive data from them
    fn broadcast_many<F: CanonicalSerialize + CanonicalDeserialize>(
        &mut self,
        data: &[F],
    ) -> std::io::Result<(Vec<F>, Vec<F>)>;

    /// Sends data to the target party. This function has a default implementation for calling [Rep3Network::send_many].
    fn send<F: CanonicalSerialize>(&mut self, target: PartyID, data: F) -> std::io::Result<()> {
        self.send_many(target, &[data])
    }

    /// Sends a vector of data to the target party.
    fn send_many<F: CanonicalSerialize>(
        &mut self,
        target: PartyID,
        data: &[F],
    ) -> std::io::Result<()>;

    /// Sends data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send] with the next_id.
    fn send_next<F: CanonicalSerialize>(&mut self, data: F) -> std::io::Result<()> {
        self.send(self.get_id().next_id(), data)
    }

    /// Sends a vector data to the party with id = next_id (i.e., my_id + 1 mod 3). This function has a default implementation for calling [Rep3Network::send_many] with the next_id.
    fn send_next_many<F: CanonicalSerialize>(&mut self, data: &[F]) -> std::io::Result<()> {
        self.send_many(self.get_id().next_id(), data)
    }

    /// Receives data from the party with the given id. This function has a default implementation for calling [Rep3Network::recv_many] and checking for the correct length of 1.
    fn recv<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<F> {
        let mut res = self.recv_many(from)?;
        if res.len() != 1 {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Expected 1 element, got more",
            ))
        } else {
            Ok(res.pop().unwrap())
        }
    }

    /// Receives a vector of data from the party with the given id.
    fn recv_many<F: CanonicalDeserialize>(&mut self, from: PartyID) -> std::io::Result<Vec<F>>;

    /// Receives data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv] with the prev_id.
    fn recv_prev<F: CanonicalDeserialize>(&mut self) -> std::io::Result<F> {
        self.recv(self.get_id().prev_id())
    }

    /// Receives a vector of data from the party with the id = prev_id (i.e., my_id + 2 mod 3). This function has a default implementation for calling [Rep3Network::recv_many] with the prev_id.
    fn recv_prev_many<F: CanonicalDeserialize>(&mut self) -> std::io::Result<Vec<F>> {
        self.recv_many(self.get_id().prev_id())
    }

    /// Fork the network into two separate instances with their own connections
    fn fork(&mut self) -> std::io::Result<Self>
    where
        Self: Sized;
}
